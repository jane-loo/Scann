"""Chat RAG 检索：单库 ANN + 联合索引。"""
from __future__ import annotations

import json

from ..models import AnnIndex
from ..index.manager import index_is_usable
from ..search.engine import SearchEngine
from ..search.joint_engine import pick_ready_joint_index, joint_search
from .cells import filter_cells, compute_stats


_INDEX_PRIORITY = ('hnsw', 'ivf_flat', 'ivf_pq', 'exact')


def pick_ready_index(dataset_id: int, preferred_id: int | None = None) -> AnnIndex | None:
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


def _pack_single(res: dict, ann_index: AnnIndex, search_mode: str, seed_cell_id=None, total=None) -> dict:
    results = res['results']
    return {
        'results': results,
        'total_matched': total if total is not None else len(results),
        'found': len(results),
        'search_mode': search_mode,
        'index_type': ann_index.index_type,
        'index_id': ann_index.id,
        'query_time_ms': res.get('query_time_ms', 0),
        'seed_cell_id': seed_cell_id,
        'cell_stats': compute_stats(results),
        'is_joint': False,
        'dataset_ids': None,
    }


def rag_retrieve(
    dataset_id: int,
    intent: dict,
    *,
    index_id: int | None = None,
    joint_index_id: int | None = None,
    use_joint: bool = False,
    user_id: int | None = None,
) -> dict:
    mode = (intent.get('search_mode') or 'metadata').lower()
    filters = intent.get('filters') or {}
    top_k = max(1, min(int(intent.get('top_k', 10)), 200))
    cell_id = intent.get('cell_id')

    if use_joint or intent.get('use_joint'):
        joint_idx = pick_ready_joint_index(joint_index_id, dataset_id)
        if joint_idx and mode in ('similarity', 'hybrid'):
            try:
                seed_id = cell_id
                if not seed_id and filters and mode in ('similarity', 'hybrid'):
                    seeds, _ = filter_cells(dataset_id, filters, top_k=1)
                    if seeds:
                        seed_id = seeds[0]['cell_id']
                if seed_id:
                    out = joint_search(
                        joint_idx,
                        cell_id=seed_id,
                        top_k=top_k,
                        filters=filters if mode == 'hybrid' else None,
                    )
                    stats = compute_stats(out['results'])
                    return {
                        **out,
                        'search_mode': 'joint_vector_ann' if cell_id else 'joint_vector_ann_seeded',
                        'seed_cell_id': seed_id,
                        'cell_stats': stats,
                    }
            except ValueError:
                pass

    ann_index = pick_ready_index(dataset_id, index_id)
    engine = SearchEngine()

    if mode in ('similarity', 'hybrid') and ann_index:
        try:
            if cell_id:
                res = engine.search_by_cell_id(
                    dataset_id, cell_id, ann_index.id, top_k,
                    user_id=user_id,
                    filters=filters if mode == 'hybrid' else None,
                )
                return _pack_single(res, ann_index, 'vector_ann', cell_id)

            if filters:
                seeds, total = filter_cells(dataset_id, filters, top_k=1)
                if seeds:
                    seed_id = seeds[0]['cell_id']
                    res = engine.search_by_cell_id(
                        dataset_id, seed_id, ann_index.id, top_k,
                        user_id=user_id,
                        filters=filters if mode == 'hybrid' else None,
                    )
                    return _pack_single(res, ann_index, 'vector_ann_seeded', seed_id, total)
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
        'is_joint': False,
        'dataset_ids': None,
    }
