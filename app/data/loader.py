import os
import numpy as np
import scipy.sparse as sp
import scanpy as sc

# 内存缓存：dataset_id -> dict
_dataset_cache: dict = {}

# 前端过滤时常用的 obs 列（优先展示）
_KEY_OBS_COLS = ['cell_type', 'disease', 'AgeGroup', 'donor_id',
                 'sex', 'tissue', 'author_cell_type', 'Phase']


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


def load_dataset(file_path: str) -> dict:
    """
    读取 .h5ad 文件，提取向量和元数据。

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
    """
    adata = sc.read_h5ad(file_path)

    # 向量优先级：X_pca > 原始矩阵
    if 'X_pca' in adata.obsm:
        vectors = np.array(adata.obsm['X_pca'], dtype=np.float32)
    else:
        X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
        vectors = X.astype(np.float32)

    obs      = adata.obs
    cell_ids = adata.obs_names.tolist()

    cell_types  = (obs['cell_type'].dropna().unique().tolist()
                   if 'cell_type' in obs.columns else [])
    obs_columns = obs.columns.tolist()

    umap_coords = (np.array(adata.obsm['X_umap'], dtype=np.float32)
                   if 'X_umap' in adata.obsm else None)
    tsne_coords = (np.array(adata.obsm['X_tsne'], dtype=np.float32)
                   if 'X_tsne' in adata.obsm else None)

    return {
        'vectors':     vectors,
        'cell_ids':    cell_ids,
        'obs':         obs,
        'n_cells':     adata.n_obs,
        'n_genes':     adata.n_vars,
        'n_dims':      vectors.shape[1],
        'cell_types':  cell_types,
        'obs_columns': obs_columns,
        'umap_coords': umap_coords,
        'tsne_coords': tsne_coords,
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
