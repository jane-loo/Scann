import json
import numpy as np
import pandas as pd
from flask import request, jsonify
from . import chat_bp
from ..models import db, Dataset, AnnIndex
from ..data.loader import _dataset_cache, load_dataset, cache_dataset
from ..permissions import get_accessible_dataset


def _ensure_data(dataset_id: int):
    if dataset_id not in _dataset_cache:
        ds = db.session.get(Dataset, dataset_id)
        if not ds:
            raise ValueError(f'数据集 {dataset_id} 不存在')
        data = load_dataset(ds.file_path)
        cache_dataset(dataset_id, data)
    return _dataset_cache[dataset_id]


def _filter_cells(dataset_id: int, filters: dict, top_k: int) -> tuple:
    """按 metadata 过滤，返回 (results_list, total_matched)。"""
    data = _ensure_data(dataset_id)
    obs: pd.DataFrame = data['obs']
    cell_ids: list = data['cell_ids']

    mask = pd.Series([True] * len(obs), index=obs.index)

    for key, vals in filters.items():
        if key not in obs.columns:
            continue
        if isinstance(vals, str):
            vals = [vals]
        col_str = obs[key].astype(str).str.lower()
        # 先精确匹配，失败则模糊包含匹配
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


def _compute_stats(results: list) -> dict:
    stats: dict = {}
    for cell in results:
        for key, val in cell.get('metadata', {}).items():
            stats.setdefault(key, {})
            stats[key][str(val)] = stats[key].get(str(val), 0) + 1
    return stats


@chat_bp.route('/query', methods=['POST'])
def chat_query():
    body = request.get_json() or {}
    message    = body.get('message', '').strip()
    dataset_id = body.get('dataset_id')
    history    = body.get('history', [])

    if not message:
        return jsonify({'error': '消息不能为空'}), 400
    if not dataset_id:
        return jsonify({'error': '请先选择数据集'}), 400

    _, err, code = get_accessible_dataset(int(dataset_id))
    if err:
        return err, code

    ds = db.session.get(Dataset, int(dataset_id))
    if not ds:
        return jsonify({'error': '数据集不存在'}), 404

    obs_columns = json.loads(ds.obs_columns) if ds.obs_columns else []

    try:
        from .llm_client import parse_intent, analyze_results
        from .knowledge import collect_knowledge

        # 1. 解析意图
        intent = parse_intent(message, obs_columns, history)
        filters = intent.get('filters', {}) or {}
        top_k   = max(1, min(int(intent.get('top_k', 10)), 200))

        # 2. metadata 过滤
        results, total_matched = _filter_cells(int(dataset_id), filters, top_k)

        # 3. 统计分布
        cell_stats = _compute_stats(results)

        # 4. 查知识库
        knowledge = collect_knowledge(filters)

        # 5. LLM 分析
        analysis = analyze_results(message, intent, results, cell_stats, knowledge, history)

        return jsonify({
            'intent':          intent,
            'filters_applied': filters,
            'top_k':           top_k,
            'total_matched':   total_matched,
            'found':           len(results),
            'cell_stats':      cell_stats,
            'knowledge':       knowledge,
            'analysis':        analysis,
            'results':         results[:50],
        })

    except RuntimeError as e:
        # API key 未配置等
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'分析失败: {str(e)}'}), 500
