from flask import request, jsonify
from flask_login import login_required, current_user
from .engine import SearchEngine
from . import search_bp

engine = SearchEngine()

@search_bp.route('/by_cell_id', methods=['POST'])
@login_required
def search_by_cell_id():
    """按 Cell ID 检索相似细胞"""
    data = request.get_json() or {}
    dataset_id = data.get('dataset_id')
    cell_id    = data.get('cell_id')
    index_id   = data.get('index_id')
    top_k      = data.get('top_k', 10)

    if not all([dataset_id, cell_id, index_id]):
        return jsonify({'error': '缺少必要参数 (dataset_id, cell_id, index_id)'}), 400

    try:
        res = engine.search_by_cell_id(int(dataset_id), cell_id, int(index_id), 
                                      top_k=int(top_k), user_id=current_user.id)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@search_bp.route('/by_vector', methods=['POST'])
@login_required
def search_by_vector():
    """按向量检索相似细胞"""
    data = request.get_json() or {}
    dataset_id = data.get('dataset_id')
    vector     = data.get('vector')
    index_id   = data.get('index_id')
    top_k      = data.get('top_k', 10)

    if not all([dataset_id, vector, index_id]):
        return jsonify({'error': '缺少必要参数 (dataset_id, vector, index_id)'}), 400

    try:
        res = engine.search_by_vector(int(dataset_id), vector, int(index_id), 
                                     top_k=int(top_k), user_id=current_user.id)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@search_bp.route('/random', methods=['POST'])
@login_required
def search_random():
    """随机选择一个细胞进行检索（用于演示）"""
    data = request.get_json() or {}
    dataset_id = data.get('dataset_id')
    index_id   = data.get('index_id')
    top_k      = data.get('top_k', 10)

    if not all([dataset_id, index_id]):
        return jsonify({'error': '缺少必要参数 (dataset_id, index_id)'}), 400

    try:
        res = engine.search_random(int(dataset_id), int(index_id), 
                                  top_k=int(top_k), user_id=current_user.id)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
