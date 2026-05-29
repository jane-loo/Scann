"""
ANN 索引统一管理器。

职责：
  1. 异步/同步构建索引，完成后将 status / build_time / index_file 写回 DB
  2. 按需从文件加载索引到内存缓存（_index_cache）
  3. 提供统一的 search_index() 入口，屏蔽 HNSW / FAISS 差异
  4. 提供 evict_index_cache() 供删除时清理缓存
"""
import json
import os
import threading
import time

import numpy as np

from ..models import db, AnnIndex, Dataset
from ..data.loader import _dataset_cache, load_dataset, cache_dataset
from .hnsw_index  import HNSWIndex
from .faiss_index import FaissIndex

# ── 内存索引缓存 ──────────────────────────────────────────────────────────────
# { ann_index_id(int): HNSWIndex | FaissIndex }
_index_cache: dict = {}
_cache_lock         = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _index_file_path(index_folder: str,
                     dataset_id:   int,
                     ann_index_id: int,
                     index_type:   str) -> str:
    ext = '.bin' if index_type == 'hnsw' else '.faiss'
    return os.path.join(
        index_folder,
        f'ds{dataset_id}_idx{ann_index_id}_{index_type}{ext}',
    )


def _ensure_vectors(dataset_id: int) -> np.ndarray:
    """确保数据集向量在缓存中，返回 vectors ndarray。"""
    if dataset_id not in _dataset_cache:
        dataset = db.session.get(Dataset, dataset_id)
        if dataset is None:
            raise RuntimeError(f'数据集 {dataset_id} 不存在')
        data = load_dataset(dataset.file_path)
        cache_dataset(dataset_id, data)
    return _dataset_cache[dataset_id]['vectors']


# ──────────────────────────────────────────────────────────────────────────────
# 构建（同步，在独立线程中调用）
# ──────────────────────────────────────────────────────────────────────────────

def build_index_sync(app,
                     dataset_id:   int,
                     ann_index_id: int,
                     index_type:   str,
                     metric:       str,
                     params:       dict,
                     index_folder: str) -> None:
    """
    同步构建索引，应在独立后台线程中调用。
    构建完成（或失败）后将结果写回数据库。
    """
    with app.app_context():
        ann_index = db.session.get(AnnIndex, ann_index_id)
        if ann_index is None:
            return

        try:
            t0      = time.time()
            vectors = _ensure_vectors(dataset_id)

            os.makedirs(index_folder, exist_ok=True)
            index_path = _index_file_path(
                index_folder, dataset_id, ann_index_id, index_type
            )

            if index_type == 'hnsw':
                idx = HNSWIndex()
                idx.build(
                    vectors,
                    metric          = metric,
                    M               = params.get('M', 16),
                    ef_construction = params.get('ef_construction', 200),
                )
                idx.save(index_path)

            else:
                # exact / ivf_flat / ivf_pq
                idx = FaissIndex()
                idx.build(
                    vectors,
                    index_type = index_type,
                    metric     = metric,
                    nlist      = params.get('nlist', 100),
                    m_pq       = params.get('m_pq', 8),
                    nbits      = params.get('nbits', 8),
                )
                idx.save(index_path)

            with _cache_lock:
                _index_cache[ann_index_id] = idx

            elapsed = time.time() - t0
            ann_index.status     = 'ready'
            ann_index.build_time = round(elapsed, 3)
            ann_index.index_file = index_path
            db.session.commit()

        except Exception as exc:
            db.session.rollback()  # 必须回滚，否则 session 会被污染
            print(f'[错误] 索引 {ann_index_id} 构建失败: {exc}')
            
            # 重新获取对象以规避之前的 PendingRollbackError
            try:
                # 使用 db.session.get 重新加载对象
                ann_index = db.session.get(AnnIndex, ann_index_id)
                if ann_index:
                    ann_index.status = 'error'
                    existing = json.loads(ann_index.params or '{}')
                    existing['_error'] = str(exc)
                    ann_index.params = json.dumps(existing, ensure_ascii=False)
                    db.session.commit()
            except Exception as e2:
                print(f'[严重错误] 无法保存错误状态到数据库: {e2}')
                db.session.rollback()
            raise


# ──────────────────────────────────────────────────────────────────────────────
# 异步启动
# ──────────────────────────────────────────────────────────────────────────────

def build_index_async(app,
                      dataset_id:   int,
                      ann_index_id: int,
                      index_type:   str,
                      metric:       str,
                      params:       dict,
                      index_folder: str) -> threading.Thread:
    """
    启动后台守护线程执行索引构建，立即返回线程对象。
    调用方无需等待。
    """
    t = threading.Thread(
        target  = build_index_sync,
        args    = (app, dataset_id, ann_index_id,
                   index_type, metric, params, index_folder),
        daemon  = True,
        name    = f'index-build-{ann_index_id}',
    )
    t.start()
    return t


# ──────────────────────────────────────────────────────────────────────────────
# 从文件加载索引
# ──────────────────────────────────────────────────────────────────────────────

def _load_index_from_file(ann_index: AnnIndex):
    """按类型从文件恢复索引对象（不写缓存，由调用方决定）。"""
    if not ann_index.index_file or not os.path.exists(ann_index.index_file):
        raise RuntimeError(
            f'索引文件不存在: {ann_index.index_file}，请重新构建'
        )

    if ann_index.index_type == 'hnsw':
        params = json.loads(ann_index.params or '{}')
        dim    = params.get('dim', 30)
        space  = 'cosine' if ann_index.metric == 'cosine' else 'l2'
        idx = HNSWIndex()
        idx.load(ann_index.index_file, dim=dim, space=space)
    else:
        idx = FaissIndex()
        idx.load(ann_index.index_file, metric=ann_index.metric)

    return idx


# ──────────────────────────────────────────────────────────────────────────────
# 搜索
# ──────────────────────────────────────────────────────────────────────────────

def search_index(ann_index:    AnnIndex,
                 query_vector: np.ndarray,
                 k:            int = 10,
                 nprobe:       int = 10) -> tuple[list[int], list[float]]:
    """
    对已构建的索引执行 kNN 搜索。

    Returns:
        (indices, distances)：整数位置列表 + 距离列表
        与 _dataset_cache[dataset_id]['cell_ids'] 下标对应。
    """
    with _cache_lock:
        idx = _index_cache.get(ann_index.id)

    if idx is None:
        idx = _load_index_from_file(ann_index)
        with _cache_lock:
            _index_cache[ann_index.id] = idx

    if ann_index.index_type == 'hnsw':
        return idx.search(query_vector, k=k)
    else:
        return idx.search(query_vector, k=k, nprobe=nprobe)


# ──────────────────────────────────────────────────────────────────────────────
# 供外部模块（Member B）直接使用的高层入口
# ──────────────────────────────────────────────────────────────────────────────

def get_index_object(index_id: int):
    """
    按 index_id 获取已加载的索引对象（HNSWIndex 或 FaissIndex）。
    若不在内存缓存中，则自动从文件加载。
    若索引状态不为 ready 或文件不存在，返回 None。

    供 Member B 检索模块调用：
        from app.index import get_index_object
        idx = get_index_object(index_id)
        indices, distances = idx.search(query_vec, k=10)
    """
    with _cache_lock:
        obj = _index_cache.get(index_id)
    if obj is not None:
        return obj

    # 从数据库查记录，再从文件加载
    ann_index = db.session.get(AnnIndex, index_id)
    if ann_index is None or ann_index.status != 'ready':
        return None

    try:
        obj = _load_index_from_file(ann_index)
    except RuntimeError:
        return None

    with _cache_lock:
        _index_cache[index_id] = obj
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# 7.3 多数据集联合索引
# ──────────────────────────────────────────────────────────────────────────────

def _joint_mapping_path(index_path: str) -> str:
    """联合索引的映射文件路径（与索引文件同目录，扩展名换为 .map.json）。"""
    return os.path.splitext(index_path)[0] + '.map.json'


def build_joint_index_sync(app,
                           ann_index_id: int,
                           index_type:   str,
                           metric:       str,
                           params:       dict,
                           index_folder: str) -> None:
    """
    同步构建联合索引（多数据集向量合并），在后台线程中调用。

    params 中必须包含 'dataset_ids': list[int]。
    构建完成后写回 DB，并将映射文件路径存入 params['mapping_file']。
    """
    with app.app_context():
        ann_index = db.session.get(AnnIndex, ann_index_id)
        if ann_index is None:
            return

        try:
            t0 = time.time()
            dataset_ids: list = params.get('dataset_ids', [])
            if not dataset_ids:
                raise RuntimeError('params 中缺少 dataset_ids')

            # ── 1. 加载各数据集向量，构建全局映射表 ────────────────────────
            all_vectors: list[np.ndarray] = []
            mapping: list[dict] = []   # [{'dataset_id', 'cell_id', 'pos_in_dataset'}]

            for ds_id in dataset_ids:
                dataset = db.session.get(Dataset, ds_id)
                if dataset is None:
                    raise RuntimeError(f'数据集 {ds_id} 不存在')
                if ds_id not in _dataset_cache:
                    data = load_dataset(dataset.file_path)
                    cache_dataset(ds_id, data)
                data = _dataset_cache[ds_id]
                for pos, cell_id in enumerate(data['cell_ids']):
                    mapping.append({
                        'dataset_id':     ds_id,
                        'cell_id':        cell_id,
                        'pos_in_dataset': pos,
                    })
                all_vectors.append(data['vectors'])

            combined = np.vstack(all_vectors).astype(np.float32)

            # ── 2. 构建索引并保存 ────────────────────────────────────────────
            os.makedirs(index_folder, exist_ok=True)
            # dataset_id=0 用作联合索引的文件名前缀，不对应任何真实数据集
            index_path   = _index_file_path(index_folder, 0, ann_index_id, index_type)
            mapping_path = _joint_mapping_path(index_path)

            if index_type == 'hnsw':
                idx = HNSWIndex()
                idx.build(combined, metric=metric,
                          M=params.get('M', 16),
                          ef_construction=params.get('ef_construction', 200))
                idx.save(index_path)
            else:
                idx = FaissIndex()
                idx.build(combined, index_type=index_type, metric=metric,
                          nlist=params.get('nlist', 100),
                          m_pq=params.get('m_pq', 8),
                          nbits=params.get('nbits', 8))
                idx.save(index_path)

            # ── 3. 保存映射文件 ──────────────────────────────────────────────
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, ensure_ascii=False)

            with _cache_lock:
                _index_cache[ann_index_id] = idx

            elapsed = time.time() - t0

            # ── 4. 写回数据库 ────────────────────────────────────────────────
            updated = dict(params)
            updated['mapping_file'] = mapping_path
            updated['total_cells']  = len(mapping)
            updated['dim']          = combined.shape[1]

            ann_index.status     = 'ready'
            ann_index.build_time = round(elapsed, 3)
            ann_index.index_file = index_path
            ann_index.params     = json.dumps(updated, ensure_ascii=False)
            db.session.commit()

        except Exception as exc:
            ann_index.status = 'error'
            existing = json.loads(ann_index.params or '{}')
            existing['_error'] = str(exc)
            ann_index.params = json.dumps(existing, ensure_ascii=False)
            db.session.commit()
            raise


def build_joint_index_async(app,
                            ann_index_id: int,
                            index_type:   str,
                            metric:       str,
                            params:       dict,
                            index_folder: str) -> threading.Thread:
    """启动后台线程执行联合索引构建，立即返回。"""
    t = threading.Thread(
        target  = build_joint_index_sync,
        args    = (app, ann_index_id, index_type, metric, params, index_folder),
        daemon  = True,
        name    = f'joint-index-build-{ann_index_id}',
    )
    t.start()
    return t


def search_joint_index(ann_index:    AnnIndex,
                       query_vector: np.ndarray,
                       k:            int = 10,
                       nprobe:       int = 10) -> list[dict]:
    """
    在联合索引上执行 kNN 搜索，将位置编号映射回 (dataset_id, cell_id)。

    Returns:
        list of {'dataset_id', 'cell_id', 'distance', 'pos_in_dataset'}
    """
    params       = json.loads(ann_index.params or '{}')
    mapping_path = params.get('mapping_file')

    if not mapping_path or not os.path.exists(mapping_path):
        raise RuntimeError('联合索引映射文件不存在，请重新构建')

    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping: list[dict] = json.load(f)

    indices, distances = search_index(ann_index, query_vector, k=k, nprobe=nprobe)

    results = []
    for pos, dist in zip(indices, distances):
        if 0 <= pos < len(mapping):
            entry = mapping[pos]
            results.append({
                'dataset_id':     entry['dataset_id'],
                'cell_id':        entry['cell_id'],
                'pos_in_dataset': entry['pos_in_dataset'],
                'distance':       float(dist),
            })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 缓存清理
# ──────────────────────────────────────────────────────────────────────────────

def evict_index_cache(ann_index_id: int) -> None:
    """从内存缓存中移除指定索引（删除索引时调用）。"""
    with _cache_lock:
        _index_cache.pop(ann_index_id, None)
