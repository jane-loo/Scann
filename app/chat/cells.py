"""Chat 模块：按 metadata 过滤细胞。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Dataset, db
from ..data.loader import _dataset_cache, load_dataset, cache_dataset


def ensure_data(dataset_id: int):
    if dataset_id not in _dataset_cache:
        ds = db.session.get(Dataset, dataset_id)
        if not ds:
            raise ValueError(f'数据集 {dataset_id} 不存在')
        data = load_dataset(ds.file_path)
        cache_dataset(dataset_id, data)
    return _dataset_cache[dataset_id]


def filter_cells(dataset_id: int, filters: dict, top_k: int) -> tuple[list, int]:
    """按 metadata 过滤，返回 (results_list, total_matched)。"""
    data = ensure_data(dataset_id)
    obs: pd.DataFrame = data['obs']
    cell_ids: list = data['cell_ids']

    mask = pd.Series([True] * len(obs), index=obs.index)

    for key, vals in filters.items():
        if key not in obs.columns:
            continue
        if isinstance(vals, str):
            vals = [vals]
        col_str = obs[key].astype(str).str.lower()
        exact_mask = col_str.isin([v.lower() for v in vals])
        if exact_mask.any():
            mask = mask & exact_mask
        else:
            pattern = '|'.join(v.lower() for v in vals)
            mask = mask & col_str.str.contains(pattern, na=False)

    positions = np.where(mask.values)[0]
    total = len(positions)

    results = []
    for rank, pos in enumerate(positions[:top_k], start=1):
        row = obs.iloc[pos]
        cell_id = cell_ids[pos]
        results.append({
            'rank': rank,
            'cell_id': cell_id,
            'cell_index': cell_id,
            'distance': 0.0,
            'similarity': 1.0,
            'cell_type': str(row.get('cell_type', '未知类型')),
            'metadata': {
                col: str(row[col])
                for col in obs.columns
                if not pd.isna(row[col])
            },
        })

    return results, total


def compute_stats(results: list) -> dict:
    stats: dict = {}
    for cell in results:
        for key, val in cell.get('metadata', {}).items():
            stats.setdefault(key, {})
            stats[key][str(val)] = stats[key].get(str(val), 0) + 1
    return stats
