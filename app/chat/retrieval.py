"""Chat RAG 检索：结合 ANN 向量检索与 metadata 过滤。"""
from __future__ import annotations

import json

from ..models import AnnIndex
from ..index.manager import index_is_usable
from ..search.engine import SearchEngine
from .cells import filter_cells, compute_stats


_INDEX_PRIORITY = ('hnsw', 'ivf_flat', 'ivf_pq', 'exact')


def pick_ready_index(dataset_id: int, preferred_id: int | None = None) -> AnnIndex | None:
    """为数据集选择可用的 ANN 索引（优先 hnsw，跳过联合索引）。"""
    if preferred_id:
        idx = AnnIndex.query.get(preferred_id)
        if (
            idx
            and idx.dataset_id == dataset_id
            and index_is_usable(idx)
            and not json.loads(idx.params or '{}').get('joint')
        ):
            return idx

    for index_type in _INDEX_PRIORITY:
        for idx in AnnIndex.query.filter_by(dataset_id=dataset_id, index_type=index_type).all():
            if index_is_usable(idx) and not json.loads(idx.params or '{}').get('joint'):
                return idx
    return None


def rag_retrieve(
    dataset_id: int,
    intent: dict,
    *,
    index_id: int | None = None,
    user_id: int | None = None,
) -> dict:
    """
    根据 LLM 解析意图执行检索。

    search_mode:
      - metadata: 仅 metadata 过滤
      - similarity: ANN 向量检索（需 cell_id 或 filters 作种子）
      - hybrid: ANN + metadata 后过滤
    """
    mode = (intent.get('search_mode') or 'metadata').lower()
    filters = intent.get('filters') or {}
    top_k = max(1, min(int(intent.get('top_k', 10)), 200))
    cell_id = intent.get('cell_id')

    ann_index = pick_ready_index(dataset_id, index_id)
    engine = SearchEngine()

    if mode in ('similarity', 'hybrid') and ann_index:
        try:
            if cell_id:
                res = engine.search_by_cell_id(
                    dataset_id,
                    cell_id,
                    ann_index.id,
                    top_k,
                    user_id=user_id,
                    filters=filters if mode == 'hybrid' else None,
                )
                return {
                    'results': res['results'],
                    'total_matched': len(res['results']),
                    'found': len(res['results']),
                    'search_mode': 'vector_ann',
                    'index_type': ann_index.index_type,
                    'index_id': ann_index.id,
                    'query_time_ms': res.get('query_time_ms', 0),
                    'seed_cell_id': cell_id,
                    'cell_stats': compute_stats(res['results']),
                }

            if filters:
                seeds, total = filter_cells(dataset_id, filters, top_k=1)
                if seeds:
                    seed_id = seeds[0]['cell_id']
                    res = engine.search_by_cell_id(
                        dataset_id,
                        seed_id,
                        ann_index.id,
                        top_k,
                        user_id=user_id,
                        filters=filters if mode == 'hybrid' else None,
                    )
                    return {
                        'results': res['results'],
                        'total_matched': total,
                        'found': len(res['results']),
                        'search_mode': 'vector_ann_seeded',
                        'index_type': ann_index.index_type,
                        'index_id': ann_index.id,
                        'query_time_ms': res.get('query_time_ms', 0),
                        'seed_cell_id': seed_id,
                        'cell_stats': compute_stats(res['results']),
                    }
        except ValueError:
            pass

    results, total = filter_cells(dataset_id, filters, top_k)
    return {
        'results': results,
        'total_matched': total,
        'found': len(results),
        'search_mode': 'metadata',
        'index_type': None,
        'index_id': None,
        'query_time_ms': 0,
        'seed_cell_id': None,
        'cell_stats': compute_stats(results),
    }
