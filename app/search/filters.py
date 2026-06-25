"""元数据过滤与 ANN 检索扩 k 的共享逻辑。"""

from __future__ import annotations

import pandas as pd


def compute_expanded_k(
    top_k: int,
    *,
    exclude_self: bool = False,
    filters: dict | None = None,
    max_k: int = 2000,
    multiplier: int = 10,
) -> int:
    """有过滤或排除自身时扩大候选检索数量。"""
    if exclude_self or filters:
        return min(top_k * multiplier, max_k)
    return top_k


def row_matches_filters(row, filters: dict | None) -> bool:
    """检查单行 obs 是否满足 filters（AND，值支持单值或列表）。"""
    if not filters:
        return True

    for key, val_list in filters.items():
        if key not in row.index and key not in row:
            return False

        target_val = str(row[key])
        if isinstance(val_list, list):
            if target_val not in [str(v) for v in val_list]:
                return False
        elif target_val != str(val_list):
            return False

    return True


def build_result_metadata(row, obs: pd.DataFrame) -> dict:
    """提取一行 obs 的全部非空元数据。"""
    return {
        col: str(row[col])
        for col in obs.columns
        if not pd.isna(row[col])
    }
