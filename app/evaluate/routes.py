import time
import json
import numpy as np
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from ..models import db, Dataset, AnnIndex, EvaluationReport
from ..decorators import expert_required
from ..index.manager import search_index, _ensure_vectors
from .metrics import recall_at_k, compute_qps
from . import evaluate_bp

@evaluate_bp.route('/<int:dataset_id>', methods=['POST'])
@login_required
@expert_required
def trigger_evaluate(dataset_id):
    """触发性能评测 (仅限 expert 以上角色)"""
    data_json = request.get_json() or {}
    index_id  = data_json.get('index_id')
    k         = data_json.get('k', 10)
    n_queries = data_json.get('n_queries', 50) # 默认评测 50 个点

    if not index_id:
        return jsonify({'error': '缺少 index_id'}), 400

    # 1. 检查 dataset 和 index
    dataset = db.session.get(Dataset, dataset_id)
    target_index = db.session.get(AnnIndex, index_id)
    if not dataset or not target_index:
        return jsonify({'error': '数据集或索引不存在'}), 404
    
    if target_index.status != 'ready':
        return jsonify({'error': '索引未就绪'}), 400

    # 2. 寻找该数据集的 Exact 索引作为 Ground Truth (必须有文件路径)
    gt_index = AnnIndex.query.filter(
        AnnIndex.dataset_id == dataset_id,
        AnnIndex.index_type == 'exact',
        AnnIndex.status == 'ready',
        AnnIndex.index_file.isnot(None)
    ).first()
    if not gt_index:
        return jsonify({'error': '未找到物理存在的精确索引(exact)，请先成功构建一次 Exact 索引'}), 400

    try:
        # 3. 准备评测数据
        vectors = _ensure_vectors(dataset_id)
        n_total = len(vectors)
        if n_total < n_queries:
            n_queries = n_total
        
        # 随机采样查询点
        indices = np.random.choice(n_total, n_queries, replace=False)
        query_vectors = vectors[indices]

        # 4. 执行评测
        recalls = []
        ann_latencies = []
        exact_latencies = []

        # 获取数据缓存以获取 cell_ids
        from ..data.loader import _dataset_cache
        cell_ids = _dataset_cache[dataset_id]['cell_ids']

        for q_vec in query_vectors:
            # GT 搜索 (精确检索)
            t_ex0 = time.time()
            gt_pos, _ = search_index(gt_index, q_vec, k=k)
            exact_latencies.append((time.time() - t_ex0) * 1000)
            gt_ids = [cell_ids[i] for i in gt_pos]

            # ANN 搜索
            t_ann0 = time.time()
            ann_pos, _ = search_index(target_index, q_vec, k=k)
            ann_latencies.append((time.time() - t_ann0) * 1000)
            ann_ids = [cell_ids[i] for i in ann_pos]

            # 计算 Recall
            recalls.append(recall_at_k(ann_ids, gt_ids, k=k))

        avg_recall  = float(np.mean(recalls))
        avg_ann_latency = float(np.mean(ann_latencies))
        avg_exact_latency = float(np.mean(exact_latencies))
        
        # 计算加速比
        speedup = avg_exact_latency / avg_ann_latency if avg_ann_latency > 0 else 1.0

        # 5. 保存报告
        report = EvaluationReport(
            dataset_id  = dataset_id,
            index_id    = index_id,
            recall_at_k = avg_recall,
            qps         = 1000.0 / avg_ann_latency if avg_ann_latency > 0 else 0,
            avg_latency = avg_ann_latency,
            n_queries   = n_queries,
            k           = k
        )
        db.session.add(report)
        db.session.commit()

        return jsonify({
            'dataset_id': dataset_id,
            'index_type': target_index.index_type,
            'n_queries': n_queries,
            'recall_at_k': round(avg_recall, 4),
            'avg_ann_time_ms': round(avg_ann_latency, 2),
            'avg_exact_time_ms': round(avg_exact_latency, 2),
            'speedup': round(speedup, 2),
            'timestamp': report.created_at.isoformat()
        })

    except Exception as e:
        return jsonify({'error': f'评测失败: {str(e)}'}), 500


@evaluate_bp.route('/<int:dataset_id>/report', methods=['GET'])
@login_required
@expert_required
def get_report(dataset_id):
    """获取最新评测报告 (仅限 expert 以上角色)"""
    # 也可以按 index_id 过滤，这里默认取该数据集下最新的
    report = EvaluationReport.query.filter_by(dataset_id=dataset_id)\
                             .order_by(EvaluationReport.created_at.desc()).first()
    
    if not report:
        return jsonify({'error': '未找到评测报告'}), 404
    
    return jsonify({
        'id': report.id,
        'index_id': report.index_id,
        'index_type': report.index.index_type if report.index else 'unknown',
        'recall_at_k': report.recall_at_k,
        'qps': report.qps,
        'avg_latency_ms': report.avg_latency,
        'n_queries': report.n_queries,
        'k': report.k,
        'created_at': report.created_at.isoformat()
    })
