"""联合索引检索引擎（供 Chat RAG 与 API 复用）。"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from ..models import AnnIndex, Dataset, db
from ..data.loader import _dataset_cache, load_dataset, cache_dataset
from ..index.manager import search_joint_index, index_is_usable
from ..search.filters import compute_expanded_k, row_matches_filters, build_result_metadata


_META_COLS = ['cell_type', 'disease', 'AgeGroup', 'donor_id', 'sex']


def pick_ready_joint_index(
    preferred_id: int | None = None,
    dataset_id: int | None = None,
) -> AnnIndex | None:
    """选择可用的联合索引；若指定 dataset_id，优先覆盖该库的联合索引。"""
    if preferred_id:
        idx = AnnIndex.query.get(preferred_id)
        if idx and index_is_usable(idx) and json.loads(idx.params or '{}').get('joint'):
            return idx

    candidates = []
    for idx in AnnIndex.query.order_by(AnnIndex.created_at.desc()).all():
        if not index_is_usable(idx):
            continue
        params = json.loads(idx.params or '{}')
        if not params.get('joint'):
            continue
        if dataset_id and dataset_id not in (params.get('dataset_ids') or []):
            continue
        candidates.append(idx)
    return candidates[0] if candidates else None


def _ensure_datasets(dataset_ids: list[int]) -> None:
    for ds_id in dataset_ids:
        if ds_id not in _dataset_cache:
            ds = db.session.get(Dataset, ds_id)
            if ds:
                cache_dataset(ds_id, load_dataset(ds.file_path))


def _load_mapping(ann_index: AnnIndex) -> list[dict]:
    params = json.loads(ann_index.params or '{}')
    path = params.get('mapping_file')
    if not path or not os.path.exists(path):
        raise ValueError('联合索引映射文件不存在，请重新构建')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _resolve_query_vector(cell_id: str, dataset_ids: list[int]) -> tuple[np.ndarray, str]:
    for ds_id in dataset_ids:
        _ensure_datasets([ds_id])
        if ds_id not in _dataset_cache:
            continue
        cache = _dataset_cache[ds_id]
        if cell_id in cache['cell_ids']:
            pos = cache['cell_ids'].index(cell_id)
            return cache['vectors'][pos], f'{ds_id}:{cell_id}'
    raise ValueError(f'所有数据集中均未找到 cell_id: {cell_id}')


def joint_search(
    ann_index: AnnIndex,
    *,
    cell_id: str | None = None,
    top_k: int = 10,
    filters: dict | None = None,
    nprobe: int = 10,
) -> dict:
    """在联合索引上按 cell_id 做相似性检索。"""
    params = json.loads(ann_index.params or '{}')
    dataset_ids = params.get('dataset_ids', [])
    if not dataset_ids:
        raise ValueError('联合索引缺少 dataset_ids')

    _ensure_datasets(dataset_ids)
    if not cell_id:
        raise ValueError('联合向量检索需要 cell_id')

    query_vec, query_repr = _resolve_query_vector(cell_id, dataset_ids)
    exclude_cell_id = cell_id
    search_k = compute_expanded_k(top_k, exclude_self=True, filters=filters)

    t0 = time.time()
    raw_results = search_joint_index(ann_index, query_vec, k=search_k, nprobe=nprobe)
    elapsed_ms = round((time.time() - t0) * 1000, 2)

    results = []
    for item in raw_results:
        if exclude_cell_id and item['cell_id'] == exclude_cell_id:
            continue
        ds_id = item['dataset_id']
        pos = item['pos_in_dataset']
        row = None
        meta = {}
        if ds_id in _dataset_cache:
            obs = _dataset_cache[ds_id]['obs']
            if pos < len(obs):
                row = obs.iloc[pos]
                if not row_matches_filters(row, filters):
                    continue
                meta = build_result_metadata(row, obs)
        elif filters:
            continue

        dist = item['distance']
        results.append({
            'rank': len(results) + 1,
            'dataset_id': ds_id,
            'cell_id': item['cell_id'],
            'distance': dist,
            'similarity': 1.0 / (1.0 + float(dist)),
            'cell_type': str(row.get('cell_type', meta.get('cell_type', '未知类型'))) if row is not None else meta.get('cell_type', '未知类型'),
            'metadata': meta,
            **{col: meta[col] for col in _META_COLS if col in meta},
        })
        if len(results) >= top_k:
            break

    return {
        'results': results,
        'found': len(results),
        'total_matched': len(results),
        'query_time_ms': elapsed_ms,
        'query_input': query_repr,
        'query_cell_id': cell_id,
        'is_joint': True,
        'dataset_ids': dataset_ids,
        'index_id': ann_index.id,
        'index_type': ann_index.index_type,
    }
