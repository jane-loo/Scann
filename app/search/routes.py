from flask import request, jsonify
from flask_login import current_user
from .engine import SearchEngine
from . import search_bp
import json
from ..models import QueryHistory
from ..decorators import login_required_api
from ..permissions import get_accessible_dataset
from ..chat.llm_client import explain_search_results

engine = SearchEngine()


def _check_search_access(dataset_id: int):
    return get_accessible_dataset(int(dataset_id))


@search_bp.route('/by_cell_id', methods=['POST'])
def search_by_cell_id():
    """按 Cell ID 检索相似细胞（访客可检索公共演示数据）"""
    data = request.get_json() or {}
    dataset_id = data.get('dataset_id')
    cell_id    = data.get('cell_id')
    index_id   = data.get('index_id')
    top_k      = data.get('top_k', 10)
    filters    = data.get('filters')

    if not all([dataset_id, cell_id, index_id]):
        return jsonify({'error': '缺少必要参数 (dataset_id, cell_id, index_id)'}), 400

    _, err, code = _check_search_access(dataset_id)
    if err:
        return err, code

    user_id = current_user.id if current_user.is_authenticated else None
    try:
        res = engine.search_by_cell_id(int(dataset_id), cell_id, int(index_id),
                                      top_k=int(top_k), user_id=user_id, filters=filters)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@search_bp.route('/by_vector', methods=['POST'])
def search_by_vector():
    """按向量检索相似细胞"""
    data = request.get_json() or {}
    dataset_id = data.get('dataset_id')
    vector     = data.get('vector')
    index_id   = data.get('index_id')
    top_k      = data.get('top_k', 10)
    filters    = data.get('filters')

    if not all([dataset_id, vector, index_id]):
        return jsonify({'error': '缺少必要参数 (dataset_id, vector, index_id)'}), 400

    _, err, code = _check_search_access(dataset_id)
    if err:
        return err, code

    user_id = current_user.id if current_user.is_authenticated else None
    try:
        res = engine.search_by_vector(int(dataset_id), vector, int(index_id),
                                     top_k=int(top_k), user_id=user_id, filters=filters)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@search_bp.route('/random', methods=['POST'])
def search_random():
    """随机选择一个细胞进行检索（用于演示）"""
    data = request.get_json() or {}
    dataset_id = data.get('dataset_id')
    index_id   = data.get('index_id')
    top_k      = data.get('top_k', 10)
    filters    = data.get('filters')

    if not all([dataset_id, index_id]):
        return jsonify({'error': '缺少必要参数 (dataset_id, index_id)'}), 400

    _, err, code = _check_search_access(dataset_id)
    if err:
        return err, code

    user_id = current_user.id if current_user.is_authenticated else None
    try:
        res = engine.search_random(int(dataset_id), int(index_id),
                                  top_k=int(top_k), user_id=user_id, filters=filters)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@search_bp.route('/history', methods=['GET'])
@login_required_api
def get_history():
    """获取当前用户的查询历史"""
    limit = request.args.get('limit', 50, type=int)
    history = QueryHistory.query.filter_by(user_id=current_user.id)\
                                .order_by(QueryHistory.created_at.desc())\
                                .limit(limit).all()

    return jsonify([{
        'id': h.id,
        'dataset_id': h.dataset_id,
        'query_type': h.query_type,
        'query_input': h.query_input,
        'index_type': h.index_type,
        'top_k': h.top_k,
        'result_ids': json.loads(h.result_ids) if h.result_ids else [],
        'query_time': h.query_time,
        'created_at': h.created_at.isoformat()
    } for h in history])


@search_bp.route('/explain', methods=['POST'])
def explain_results():
    """可选：LLM 解读当前检索结果（不引入静态知识库）。"""
    data = request.get_json() or {}
    if not data.get('enabled', True):
        return jsonify({'error': '未启用生物学解释'}), 400

    results = data.get('results') or []
    if not results:
        return jsonify({'error': '无检索结果'}), 400

    try:
        text = explain_search_results(
            query_input=data.get('query_input', ''),
            query_type=data.get('query_type', 'cell_id'),
            results=results,
            is_joint=bool(data.get('is_joint')),
            filters=data.get('filters'),
        )
        return jsonify({'explanation': text})
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'解释生成失败: {str(e)}'}), 500
