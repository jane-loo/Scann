import time

import numpy as np
from flask import request, jsonify

from ..decorators import expert_required, login_required_api
from ..models import db, Dataset, AnnIndex, EvaluationReport
from ..index.manager import search_index, _ensure_vectors, index_is_usable
from .metrics import recall_at_k
from .storage import ensure_evaluation_report_schema, index_file_size_bytes, format_bytes
from . import evaluate_bp

_ANN_INDEX_TYPES = {'hnsw', 'ivf_flat', 'ivf_pq'}


def _report_to_dict(report: EvaluationReport) -> dict:
    ds = db.session.get(Dataset, report.dataset_id)
    idx = report.index
    ann_bytes = report.index_size_bytes or 0
    exact_bytes = report.exact_index_size_bytes or 0
    return {
        'id': report.id,
        'dataset_id': report.dataset_id,
        'dataset_name': ds.name if ds else None,
        'index_id': report.index_id,
        'index_type': idx.index_type if idx else 'unknown',
        'recall_at_k': report.recall_at_k,
        'qps': report.qps,
        'avg_latency_ms': report.avg_latency,
        'n_queries': report.n_queries,
        'k': report.k,
        'index_size_bytes': ann_bytes,
        'exact_index_size_bytes': exact_bytes,
        'index_size_mb': round(ann_bytes / (1024 * 1024), 3) if ann_bytes else 0,
        'exact_index_size_mb': round(exact_bytes / (1024 * 1024), 3) if exact_bytes else 0,
        'memory_ratio': round(ann_bytes / exact_bytes, 3) if ann_bytes and exact_bytes else None,
        'index_size_label': format_bytes(int(ann_bytes)),
        'exact_index_size_label': format_bytes(int(exact_bytes)),
        'created_at': report.created_at.isoformat(),
    }


def _find_exact_baseline(dataset_id: int) -> AnnIndex | None:
    """返回该数据集上可用的 exact 索引，作为 Ground Truth。"""
    for idx in AnnIndex.query.filter_by(dataset_id=dataset_id, index_type='exact').all():
        if index_is_usable(idx):
            return idx
    return None


def _run_single_benchmark(dataset_id: int, target_index: AnnIndex, k: int, n_queries: int) -> dict:
    gt_index = _find_exact_baseline(dataset_id)
    if not gt_index:
        raise RuntimeError('未找到可用的 Exact 索引，请先构建 Exact 作为准确度基线')

    if target_index.index_type not in _ANN_INDEX_TYPES:
        raise RuntimeError(
            f'请选择 ANN 索引 (hnsw / ivf_flat / ivf_pq)，当前为 {target_index.index_type}'
        )

    vectors = _ensure_vectors(dataset_id)
    n_total = len(vectors)
    n_queries = min(n_queries, n_total)

    indices = np.random.choice(n_total, n_queries, replace=False)
    query_vectors = vectors[indices]

    from ..data.loader import _dataset_cache
    cell_ids = _dataset_cache[dataset_id]['cell_ids']

    recalls = []
    ann_latencies = []
    exact_latencies = []

    for q_vec in query_vectors:
        t_ex0 = time.time()
        gt_pos, _ = search_index(gt_index, q_vec, k=k)
        exact_latencies.append((time.time() - t_ex0) * 1000)
        gt_ids = [cell_ids[i] for i in gt_pos]

        t_ann0 = time.time()
        ann_pos, _ = search_index(target_index, q_vec, k=k)
        ann_latencies.append((time.time() - t_ann0) * 1000)
        ann_ids = [cell_ids[i] for i in ann_pos]

        recalls.append(recall_at_k(ann_ids, gt_ids, k=k))

    avg_recall = float(np.mean(recalls))
    avg_ann_latency = float(np.mean(ann_latencies))
    avg_exact_latency = float(np.mean(exact_latencies))
    speedup = avg_exact_latency / avg_ann_latency if avg_ann_latency > 0 else 1.0

    ann_size = index_file_size_bytes(target_index)
    exact_size = index_file_size_bytes(gt_index)
    memory_ratio = ann_size / exact_size if ann_size and exact_size else None

    report = EvaluationReport(
        dataset_id=dataset_id,
        index_id=target_index.id,
        recall_at_k=avg_recall,
        qps=1000.0 / avg_ann_latency if avg_ann_latency > 0 else 0,
        avg_latency=avg_ann_latency,
        n_queries=n_queries,
        k=k,
        index_size_bytes=float(ann_size),
        exact_index_size_bytes=float(exact_size),
    )
    db.session.add(report)
    db.session.commit()

    return {
        'report_id': report.id,
        'dataset_id': dataset_id,
        'index_id': target_index.id,
        'index_type': target_index.index_type,
        'n_queries': n_queries,
        'recall_at_k': round(avg_recall, 4),
        'avg_ann_time_ms': round(avg_ann_latency, 2),
        'avg_exact_time_ms': round(avg_exact_latency, 2),
        'speedup': round(speedup, 2),
        'k': k,
        'index_size_bytes': ann_size,
        'exact_index_size_bytes': exact_size,
        'index_size_mb': round(ann_size / (1024 * 1024), 3),
        'exact_index_size_mb': round(exact_size / (1024 * 1024), 3),
        'memory_ratio': round(memory_ratio, 3) if memory_ratio else None,
        'index_size_label': format_bytes(ann_size),
        'exact_index_size_label': format_bytes(exact_size),
        'timestamp': report.created_at.isoformat(),
    }


@evaluate_bp.route('/reports', methods=['GET'])
@login_required_api
@expert_required
def list_reports():
    """列出评测报告（可按 dataset_id 过滤）。"""
    ensure_evaluation_report_schema()
    dataset_id = request.args.get('dataset_id', type=int)
    limit = min(100, max(1, int(request.args.get('limit', 30))))

    q = EvaluationReport.query.order_by(EvaluationReport.created_at.desc())
    if dataset_id:
        q = q.filter_by(dataset_id=dataset_id)

    reports = q.limit(limit).all()
    return jsonify([_report_to_dict(r) for r in reports])


@evaluate_bp.route('/<int:dataset_id>', methods=['POST'])
@login_required_api
@expert_required
def trigger_evaluate(dataset_id):
    """触发性能评测 (仅限 expert 以上角色)"""
    ensure_evaluation_report_schema()
    data_json = request.get_json() or {}
    index_id  = data_json.get('index_id')
    k         = data_json.get('k', 10)
    n_queries = data_json.get('n_queries', 50)

    if not index_id:
        return jsonify({'error': '缺少 index_id'}), 400

    dataset = db.session.get(Dataset, dataset_id)
    target_index = db.session.get(AnnIndex, index_id)
    if not dataset or not target_index:
        return jsonify({'error': '数据集或索引不存在'}), 404

    if target_index.dataset_id != dataset_id:
        return jsonify({'error': '索引与数据集不匹配'}), 400

    if not index_is_usable(target_index):
        return jsonify({'error': '目标索引未就绪或文件缺失，请重新构建'}), 400

    try:
        result = _run_single_benchmark(dataset_id, target_index, k, n_queries)
        return jsonify(result)

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'评测失败: {str(e)}'}), 500


@evaluate_bp.route('/<int:dataset_id>/batch', methods=['POST'])
@login_required_api
@expert_required
def trigger_batch_evaluate(dataset_id):
    """对数据集上全部 ready 的 ANN 索引批量 benchmark。"""
    ensure_evaluation_report_schema()
    data_json = request.get_json() or {}
    k = data_json.get('k', 10)
    n_queries = data_json.get('n_queries', 50)

    dataset = db.session.get(Dataset, dataset_id)
    if not dataset:
        return jsonify({'error': '数据集不存在'}), 404

    if not _find_exact_baseline(dataset_id):
        return jsonify({'error': '未找到可用的 Exact 索引，请先构建 Exact 作为准确度基线'}), 400

    ann_indexes = [
        idx for idx in AnnIndex.query.filter_by(dataset_id=dataset_id).all()
        if idx.index_type in _ANN_INDEX_TYPES and index_is_usable(idx)
    ]
    if not ann_indexes:
        return jsonify({'error': '没有可评测的 ANN 索引，请先构建 hnsw / ivf_flat / ivf_pq'}), 400

    results = []
    errors = []
    for idx in ann_indexes:
        try:
            results.append(_run_single_benchmark(dataset_id, idx, k, n_queries))
        except Exception as exc:
            db.session.rollback()
            errors.append({'index_id': idx.id, 'index_type': idx.index_type, 'error': str(exc)})

    if not results:
        return jsonify({'error': '批量评测全部失败', 'details': errors}), 500

    return jsonify({
        'dataset_id': dataset_id,
        'baseline': 'exact',
        'results': results,
        'errors': errors,
    })


@evaluate_bp.route('/<int:dataset_id>/report', methods=['GET'])
@login_required_api
@expert_required
def get_report(dataset_id):
    """获取最新评测报告 (仅限 expert 以上角色)"""
    # 也可以按 index_id 过滤，这里默认取该数据集下最新的
    report = EvaluationReport.query.filter_by(dataset_id=dataset_id)\
                             .order_by(EvaluationReport.created_at.desc()).first()
    
    if not report:
        return jsonify({'error': '未找到评测报告'}), 404

    return jsonify(_report_to_dict(report))
