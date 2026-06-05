import os
import numpy as np
import scipy.sparse as sp
import scanpy as sc

# 内存缓存：dataset_id -> dict
_dataset_cache: dict = {}

# 前端过滤时常用的 obs 列（优先展示）
_KEY_OBS_COLS = ['cell_type', 'disease', 'AgeGroup', 'donor_id',
                 'sex', 'tissue', 'author_cell_type', 'Phase']

# 无 X_pca 时自动向量化参数
DEFAULT_PCA_COMPONENTS = 30
SCANN_UNS_KEY = 'scann_vectorization'


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def validate_h5ad(file_path: str) -> tuple:
    """
    校验 .h5ad 文件合法性。
    返回 (True, None) 或 (False, 错误信息字符串)。
    """
    if not os.path.exists(file_path):
        return False, '文件不存在'
    if not file_path.lower().endswith('.h5ad'):
        return False, '文件格式不是 .h5ad'
    try:
        adata = sc.read_h5ad(file_path)
    except Exception as e:
        return False, f'文件无法读取：{e}'
    if adata.n_obs == 0:
        return False, '数据集细胞数为 0'
    if adata.n_vars == 0:
        return False, '数据集基因数为 0'
    return True, None


def _resolve_pca_components(n_obs: int, n_vars: int,
                            requested: int = DEFAULT_PCA_COMPONENTS) -> int:
    """PCA 成分数受细胞数、基因数上限约束。"""
    return max(2, min(requested, n_obs - 1, n_vars))


def _matrix_extremes(X) -> tuple[float, float]:
    if sp.issparse(X):
        return float(X.max()), float(X.min())
    arr = np.asarray(X)
    return float(arr.max()), float(arr.min())


def _compute_pca_vectors(adata, n_comps: int = DEFAULT_PCA_COMPONENTS) -> tuple[np.ndarray, dict]:
    """
    对无 obsm.X_pca 的数据执行标准单细胞向量化：
    归一化 → log1p → 高变基因 → PCA（Scanpy）。
    """
    work = adata.copy()
    n_comps = _resolve_pca_components(work.n_obs, work.n_vars, n_comps)

    max_val, min_val = _matrix_extremes(work.X)
    normalized = False
    # 已 log 归一化的矩阵通常数值较小；原始 count 则先做 normalize + log1p
    if max_val > 30 or min_val < 0:
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
        normalized = True

    n_hvg = min(2000, work.n_vars)
    hvg_applied = False
    if work.n_vars > 50:
        sc.pp.highly_variable_genes(work, n_top_genes=n_hvg, flavor='seurat', subset=True)
        hvg_applied = True

    sc.tl.pca(work, n_comps=n_comps, svd_solver='arpack', random_state=0)
    vectors = np.asarray(work.obsm['X_pca'], dtype=np.float32)

    steps = []
    if normalized:
        steps.append('normalize_total(1e4)')
        steps.append('log1p')
    else:
        steps.append('detect_log_normalized(skip normalize)')
    if hvg_applied:
        steps.append(f'highly_variable_genes({work.n_vars})')
    steps.append(f'PCA(n_comps={n_comps})')

    meta = {
        'source':       'auto_pca',
        'engine':       'scanpy',
        'n_components': n_comps,
        'n_genes_used': int(work.n_vars),
        'pipeline':     ' → '.join(steps),
        'computed_by':  'Scann 上传向量化流水线',
    }
    return vectors, meta


def _read_file_vectorization(adata, vectors: np.ndarray) -> dict:
    """读取 .h5ad 中已有的向量化说明，或推断为预置 X_pca。"""
    if SCANN_UNS_KEY in adata.uns:
        meta = dict(adata.uns[SCANN_UNS_KEY])
        meta.setdefault('n_components', int(vectors.shape[1]))
        return meta
    return {
        'source':       'file',
        'engine':       'precomputed',
        'n_components': int(vectors.shape[1]),
        'pipeline':     'obsm.X_pca（上传前已存在于 .h5ad）',
        'computed_by':  '用户 / 上游分析流程',
    }


def ensure_pca_vectors(file_path: str,
                       n_comps: int = DEFAULT_PCA_COMPONENTS,
                       persist: bool = True) -> tuple[np.ndarray, dict, bool]:
    """
    确保数据集具备可用于 ANN 的 PCA 向量。

    返回 (vectors, vectorization_meta, was_computed)。
    若本次新计算 PCA 且 persist=True，会写回 obsm.X_pca 与 uns.scann_vectorization。
    """
    adata = sc.read_h5ad(file_path)

    if 'X_pca' in adata.obsm:
        vectors = np.asarray(adata.obsm['X_pca'], dtype=np.float32)
        return vectors, _read_file_vectorization(adata, vectors), False

    vectors, meta = _compute_pca_vectors(adata, n_comps=n_comps)
    adata.obsm['X_pca'] = vectors
    adata.uns[SCANN_UNS_KEY] = meta

    if persist:
        adata.write_h5ad(file_path)

    return vectors, meta, True


def load_dataset(file_path: str, persist_pca: bool = True) -> dict:
    """
    读取 .h5ad 文件，提取向量和元数据。
    若缺少 obsm.X_pca，自动执行 Scanpy PCA 向量化（可选写回文件）。

    返回字典包含：
      vectors     np.ndarray  float32  (n_cells, n_dims)
      cell_ids    list[str]
      obs         pd.DataFrame
      n_cells     int
      n_genes     int
      n_dims      int
      cell_types  list[str]   cell_type 列的唯一值
      obs_columns list[str]   所有 obs 列名
      umap_coords np.ndarray  float32  (n_cells, 2) 或 None
      tsne_coords np.ndarray  float32  (n_cells, 2) 或 None
      vectorization dict      向量化来源与 PCA 参数说明
      pca_computed  bool      本次是否新计算 PCA
    """
    adata = sc.read_h5ad(file_path)
    pca_computed = False

    if 'X_pca' in adata.obsm:
        vectors = np.asarray(adata.obsm['X_pca'], dtype=np.float32)
        vectorization = _read_file_vectorization(adata, vectors)
    else:
        vectors, vectorization = _compute_pca_vectors(adata)
        adata.obsm['X_pca'] = vectors
        adata.uns[SCANN_UNS_KEY] = vectorization
        pca_computed = True
        if persist_pca:
            adata.write_h5ad(file_path)

    obs      = adata.obs
    cell_ids = adata.obs_names.tolist()

    cell_types  = (obs['cell_type'].dropna().unique().tolist()
                   if 'cell_type' in obs.columns else [])
    obs_columns = obs.columns.tolist()

    umap_coords = (np.array(adata.obsm['X_umap'], dtype=np.float32)
                   if 'X_umap' in adata.obsm else None)
    tsne_coords = (np.array(adata.obsm['X_tsne'], dtype=np.float32)
                   if 'X_tsne' in adata.obsm else None)

    # 无 UMAP 时用 PCA 前两维作为 2D 可视化回退
    if umap_coords is None and vectors.shape[1] >= 2:
        umap_coords = vectors[:, :2].copy()
        if pca_computed:
            vectorization = dict(vectorization)
            vectorization['viz_fallback'] = 'PCA Dim1×Dim2（无 X_umap 时的散点图回退）'

    return {
        'vectors':       vectors,
        'cell_ids':      cell_ids,
        'obs':           obs,
        'n_cells':       adata.n_obs,
        'n_genes':       adata.n_vars,
        'n_dims':        vectors.shape[1],
        'cell_types':    cell_types,
        'obs_columns':   obs_columns,
        'umap_coords':   umap_coords,
        'tsne_coords':   tsne_coords,
        'vectorization': vectorization,
        'pca_computed':  pca_computed,
    }


def cache_dataset(dataset_id: int, data: dict):
    """将 load_dataset 的结果放入内存缓存。"""
    _dataset_cache[dataset_id] = {
        'vectors':     data['vectors'],
        'cell_ids':    data['cell_ids'],
        'obs':         data['obs'],
        'umap_coords': data.get('umap_coords'),
        'tsne_coords': data.get('tsne_coords'),
    }


def get_cached_dataset(dataset_id: int) -> dict | None:
    """
    获取缓存的数据集。
    若不在缓存中，从数据库记录自动加载（需在 Flask 应用上下文中调用）。
    """
    if dataset_id not in _dataset_cache:
        _reload_from_db(dataset_id)
    return _dataset_cache.get(dataset_id)


def get_cell_vector(dataset_id: int, cell_id: str) -> np.ndarray:
    """
    根据细胞 ID 返回其 PCA 向量。
    供检索模块（成员 B）调用。
    """
    data = get_cached_dataset(dataset_id)
    if data is None:
        raise ValueError(f'数据集 {dataset_id} 未加载')
    try:
        idx = data['cell_ids'].index(cell_id)
    except ValueError:
        raise ValueError(f'细胞 "{cell_id}" 不存在于数据集 {dataset_id}')
    return data['vectors'][idx]


def get_obs_dataframe(dataset_id: int):
    """
    返回 obs 元数据 DataFrame。
    供检索引擎过滤使用（成员 B）。
    """
    data = get_cached_dataset(dataset_id)
    return data['obs'] if data else None


def get_vectors(dataset_id: int) -> np.ndarray | None:
    """
    返回完整向量矩阵 (n_cells, n_dims)。
    供索引构建和性能评测使用。
    """
    data = get_cached_dataset(dataset_id)
    return data['vectors'] if data else None


def evict_cache(dataset_id: int):
    """删除数据集时清除内存缓存。"""
    _dataset_cache.pop(dataset_id, None)


# ---------------------------------------------------------------------------
# 内部函数
# ---------------------------------------------------------------------------

def _reload_from_db(dataset_id: int):
    """Cache miss 时，从数据库找到文件路径后重新加载。"""
    try:
        from app.models import Dataset
        dataset = Dataset.query.get(dataset_id)
        if dataset and os.path.exists(dataset.file_path):
            data = load_dataset(dataset.file_path)
            cache_dataset(dataset_id, data)
    except Exception as e:
        print(f'[loader] 重新加载数据集 {dataset_id} 失败: {e}')
