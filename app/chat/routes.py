import json
from flask import request, jsonify
from flask_login import current_user
from . import chat_bp
from ..models import db, Dataset
from ..permissions import get_accessible_dataset
from .cells import compute_stats
from .llm_client import parse_intent, analyze_results
from .knowledge import collect_knowledge
from .retrieval import rag_retrieve


_SEARCH_MODE_LABELS = {
    'metadata': '元数据过滤',
    'vector_ann': 'ANN 向量检索',
    'vector_ann_seeded': 'ANN 相似性检索（种子细胞）',
}


@chat_bp.route('/query', methods=['POST'])
def chat_query():
    body = request.get_json() or {}
    message    = body.get('message', '').strip()
    dataset_id = body.get('dataset_id')
    history    = body.get('history', [])
    index_id   = body.get('index_id')

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
        intent = parse_intent(message, obs_columns, history)
        filters = intent.get('filters', {}) or {}
        top_k   = max(1, min(int(intent.get('top_k', 10)), 200))

        retrieval = rag_retrieve(
            int(dataset_id),
            intent,
            index_id=int(index_id) if index_id else None,
            user_id=user_id,
        )

        results = retrieval['results']
        cell_stats = retrieval.get('cell_stats') or compute_stats(results)
        knowledge = collect_knowledge(filters)

        analysis = analyze_results(
            message, intent, results, cell_stats, knowledge, history,
            search_mode=retrieval.get('search_mode'),
            index_type=retrieval.get('index_type'),
            query_time_ms=retrieval.get('query_time_ms', 0),
            seed_cell_id=retrieval.get('seed_cell_id'),
        )

        return jsonify({
            'intent':          intent,
            'filters_applied': filters,
            'top_k':           top_k,
            'total_matched':   retrieval.get('total_matched', len(results)),
            'found':           retrieval.get('found', len(results)),
            'cell_stats':      cell_stats,
            'knowledge':       knowledge,
            'analysis':        analysis,
            'results':         results[:50],
            'search_mode':     retrieval.get('search_mode', 'metadata'),
            'search_mode_label': _SEARCH_MODE_LABELS.get(
                retrieval.get('search_mode'), '检索'
            ),
            'index_type':      retrieval.get('index_type'),
            'index_id':        retrieval.get('index_id'),
            'query_time_ms':   retrieval.get('query_time_ms', 0),
            'seed_cell_id':    retrieval.get('seed_cell_id'),
        })

    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'分析失败: {str(e)}'}), 500
