import json
from flask import request, jsonify
from flask_login import current_user
from . import chat_bp
from ..models import db, Dataset, AnnIndex
from ..permissions import get_accessible_dataset, visible_datasets_query
from ..index.manager import index_is_usable
from .cells import compute_stats
from .llm_client import parse_intent, analyze_results
from .retrieval import rag_retrieve, pick_ready_joint_index


_SEARCH_MODE_LABELS = {
    'metadata': '元数据过滤',
    'vector_ann': 'ANN 向量检索',
    'vector_ann_seeded': 'ANN 相似性检索（种子细胞）',
    'joint_vector_ann': '联合索引 ANN 检索',
    'joint_vector_ann_seeded': '联合索引相似性检索',
}


def _joint_indexes_for_dataset(dataset_id: int) -> list[dict]:
    visible = {d.id for d in visible_datasets_query().all()}
    out = []
    for idx in AnnIndex.query.order_by(AnnIndex.created_at.desc()).all():
        params = json.loads(idx.params or '{}')
        if not params.get('joint'):
            continue
        ds_ids = params.get('dataset_ids') or []
        if not all(d in visible for d in ds_ids):
            continue
        if dataset_id and dataset_id not in ds_ids:
            continue
        if not index_is_usable(idx):
            continue
        out.append({
            'id': idx.id,
            'index_type': idx.index_type,
            'metric': idx.metric,
            'dataset_ids': ds_ids,
            'total_cells': params.get('total_cells'),
            'status': idx.status,
        })
    return out


@chat_bp.route('/joint_indexes', methods=['GET'])
def list_joint_indexes():
    dataset_id = request.args.get('dataset_id', type=int)
    if dataset_id:
        _, err, code = get_accessible_dataset(dataset_id)
        if err:
            return err, code
    return jsonify(_joint_indexes_for_dataset(dataset_id))


@chat_bp.route('/query', methods=['POST'])
def chat_query():
    body = request.get_json() or {}
    message         = body.get('message', '').strip()
    dataset_id      = body.get('dataset_id')
    history         = body.get('history', [])
    index_id        = body.get('index_id')
    joint_index_id  = body.get('joint_index_id')
    use_joint       = bool(body.get('use_joint', False))

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
    user_id = current_user.id if current_user.is_authenticated else None

    try:
        intent = parse_intent(message, obs_columns, history, use_joint=use_joint)
        if use_joint:
            intent['use_joint'] = True
        filters = intent.get('filters', {}) or {}
        top_k   = max(1, min(int(intent.get('top_k', 10)), 200))

        retrieval = rag_retrieve(
            int(dataset_id),
            intent,
            index_id=int(index_id) if index_id else None,
            joint_index_id=int(joint_index_id) if joint_index_id else None,
            use_joint=use_joint or intent.get('use_joint'),
            user_id=user_id,
        )

        results = retrieval['results']
        cell_stats = retrieval.get('cell_stats') or compute_stats(results)

        analysis = analyze_results(
            message, intent, results, cell_stats, history,
            search_mode=retrieval.get('search_mode'),
            index_type=retrieval.get('index_type'),
            query_time_ms=retrieval.get('query_time_ms', 0),
            seed_cell_id=retrieval.get('seed_cell_id'),
            is_joint=retrieval.get('is_joint', False),
            dataset_ids=retrieval.get('dataset_ids'),
        )

        sm = retrieval.get('search_mode', 'metadata')
        return jsonify({
            'intent':          intent,
            'filters_applied': filters,
            'top_k':           top_k,
            'total_matched':   retrieval.get('total_matched', len(results)),
            'found':           retrieval.get('found', len(results)),
            'cell_stats':      cell_stats,
            'analysis':        analysis,
            'results':         results[:50],
            'search_mode':     sm,
            'search_mode_label': _SEARCH_MODE_LABELS.get(sm, '检索'),
            'index_type':      retrieval.get('index_type'),
            'index_id':        retrieval.get('index_id'),
            'query_time_ms':   retrieval.get('query_time_ms', 0),
            'seed_cell_id':    retrieval.get('seed_cell_id'),
            'is_joint':        retrieval.get('is_joint', False),
            'dataset_ids':     retrieval.get('dataset_ids'),
        })

    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'分析失败: {str(e)}'}), 500
