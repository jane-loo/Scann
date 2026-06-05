"""
索引管理 Blueprint（挂载在 /api）。

端点一览：
  POST   /api/indexes/build              构建单数据集索引（异步，立即返回 202）
  POST   /api/indexes/joint_build        构建联合索引（多数据集，异步）
  GET    /api/indexes/                   列出索引（可按 dataset_id 过滤）
  GET    /api/indexes/<id>               查询索引状态/详情
  DELETE /api/indexes/<id>               删除索引（含映射文件）
  POST   /api/indexes/<id>/search        kNN 搜索（自动识别普通/联合索引）
  GET    /api/history/                   查询历史（可按 dataset_id / user_id 过滤）
"""
import json
import os
import time

import numpy as np
from flask import Blueprint, request, jsonify, current_app
from flask_login import current_user

from ..models import db, AnnIndex, Dataset, QueryHistory
from ..decorators import login_required_api
from ..permissions import visible_datasets_query, get_accessible_dataset, can_manage_data
from ..data.loader import _dataset_cache, load_dataset, cache_dataset
from .manager import (
    build_index_async, build_joint_index_async,
    search_index, search_joint_index,
    evict_index_cache, _joint_mapping_path,
    remove_index_record, remove_indexes_for_dataset,
    maintain_index_records, index_is_usable, effective_index_status,
)

index_bp = Blueprint('index', __name__)

_VALID_INDEX_TYPES = {'hnsw', 'ivf_flat', 'ivf_pq', 'exact'}
_VALID_METRICS     = {'l2', 'cosine'}

# obs 中附加到搜索结果的元数据字段
_META_COLS = ['cell_type', 'disease', 'AgeGroup', 'donor_id', 'sex']


def _can_access_index(ann_index: AnnIndex):
    return get_accessible_dataset(ann_index.dataset_id)


def _filter_visible_indexes(indexes):
    visible_ids = {d.id for d in visible_datasets_query().all()}
    return [i for i in indexes if i.dataset_id in visible_ids]


# ──────────────────────────────────────────────
# 构建索引（异步）
# ──────────────────────────────────────────────

@index_bp.route('/indexes/build', methods=['POST'])
@login_required_api
def build_index():
    if not can_manage_data():
        return jsonify({'error': '访客无权构建索引，请先登录'}), 403
    body       = request.get_json() or {}
    dataset_id = body.get('dataset_id')
    index_type = body.get('index_type', 'hnsw')
    metric     = body.get('metric', 'l2')
    params     = body.get('params', {})

    if not dataset_id:
        return jsonify({'error': '缺少 dataset_id'}), 400
    if index_type not in _VALID_INDEX_TYPES:
        return jsonify({'error': f'不支持的 index_type，可选: {sorted(_VALID_INDEX_TYPES)}'}), 400
    if metric not in _VALID_METRICS:
        return jsonify({'error': f'不支持的 metric，可选: {sorted(_VALID_METRICS)}'}), 400

    dataset = Dataset.query.get_or_404(dataset_id)
    _, err, code = get_accessible_dataset(dataset_id)
    if err:
        return err, code

    # 将 dim 写入 params，HNSW 加载时需要
    params = dict(params)
    params.setdefault('dim', dataset.n_dims or 30)

    # 同数据集同类型只保留一份索引，避免重复列表与误选旧记录
    remove_indexes_for_dataset(dataset_id, index_type)

    ann_index = AnnIndex(
        dataset_id = dataset_id,
        index_type = index_type,
        metric     = metric,
        params     = json.dumps(params, ensure_ascii=False),
        status     = 'building',
    )
    db.session.add(ann_index)
    db.session.flush()                  # 立即获取 id，避免 commit 后对象过期
    ann_index_id = ann_index.id
    db.session.commit()

    build_index_async(
        app          = current_app._get_current_object(),
        dataset_id   = dataset_id,
        ann_index_id = ann_index_id,
        index_type   = index_type,
        metric       = metric,
        params       = params,
        index_folder = current_app.config['INDEX_FOLDER'],
    )

    return jsonify({
        'message':  '索引构建已启动',
        'index_id': ann_index_id,
        'status':   'building',
    }), 202


# ──────────────────────────────────────────────
# 联合索引构建（7.3 加分项）
# ──────────────────────────────────────────────

@index_bp.route('/indexes/joint_build', methods=['POST'])
@login_required_api
def joint_build_index():
    if not can_manage_data():
        return jsonify({'error': '访客无权构建索引，请先登录'}), 403
    """
    将多个数据集的向量合并，构建统一 ANN 索引（异步，立即返回 202）。

    请求体：
      {
        "dataset_ids": [1, 2, 3],   // 至少 2 个
        "index_type": "hnsw",        // 同单数据集，默认 hnsw
        "metric":     "l2",
        "params":     {}             // 可选构建参数
      }
    """
    body       = request.get_json() or {}
    dataset_ids = body.get('dataset_ids', [])
    index_type  = body.get('index_type', 'hnsw')
    metric      = body.get('metric', 'l2')
    params      = dict(body.get('params', {}))

    if not dataset_ids or len(dataset_ids) < 2:
        return jsonify({'error': '至少需要 2 个 dataset_id'}), 400
    if index_type not in _VALID_INDEX_TYPES:
        return jsonify({'error': f'不支持的 index_type，可选: {sorted(_VALID_INDEX_TYPES)}'}), 400
    if metric not in _VALID_METRICS:
        return jsonify({'error': f'不支持的 metric，可选: {sorted(_VALID_METRICS)}'}), 400

    # 验证所有数据集存在且可访问
    for ds_id in dataset_ids:
        _, err, code = get_accessible_dataset(ds_id)
        if err:
            return err, code

    first_ds = db.session.get(Dataset, dataset_ids[0])
    params['joint']       = True
    params['dataset_ids'] = dataset_ids
    params.setdefault('dim', first_ds.n_dims or 30)

    ann_index = AnnIndex(
        dataset_id = dataset_ids[0],   # FK 引用第一个数据集
        index_type = index_type,
        metric     = metric,
        params     = json.dumps(params, ensure_ascii=False),
        status     = 'building',
    )
    db.session.add(ann_index)
    db.session.flush()                  # 立即获取 id，避免 commit 后对象过期
    ann_index_id = ann_index.id
    db.session.commit()

    build_joint_index_async(
        app          = current_app._get_current_object(),
        ann_index_id = ann_index_id,
        index_type   = index_type,
        metric       = metric,
        params       = params,
        index_folder = current_app.config['INDEX_FOLDER'],
    )

    return jsonify({
        'message':     '联合索引构建已启动',
        'index_id':    ann_index_id,
        'dataset_ids': dataset_ids,
        'status':      'building',
    }), 202


# ──────────────────────────────────────────────
# 索引列表
# ──────────────────────────────────────────────

@index_bp.route('/indexes/', methods=['GET'])
def list_indexes():
    maintain_index_records(current_app.config['INDEX_FOLDER'])
    dataset_id = request.args.get('dataset_id', type=int)
    q = AnnIndex.query.order_by(AnnIndex.created_at.desc())
    if dataset_id:
        q = q.filter_by(dataset_id=dataset_id)
    indexes = _filter_visible_indexes(q.all())
    return jsonify([_index_to_dict(i) for i in indexes])


# ──────────────────────────────────────────────
# 索引详情 / 状态轮询
# ──────────────────────────────────────────────

@index_bp.route('/indexes/<int:index_id>', methods=['GET'])
def get_index(index_id):
    db.session.expire_all()
    ann_index = AnnIndex.query.get_or_404(index_id)
    _, err, code = _can_access_index(ann_index)
    if err:
        return err, code
    return jsonify(_index_to_dict(ann_index))


# ──────────────────────────────────────────────
# 删除索引
# ──────────────────────────────────────────────

@index_bp.route('/indexes/<int:index_id>', methods=['DELETE'])
@login_required_api
def delete_index(index_id):
    if not can_manage_data():
        return jsonify({'error': '访客无权删除索引，请先登录'}), 403
    ann_index = AnnIndex.query.get_or_404(index_id)
    _, err, code = _can_access_index(ann_index)
    if err:
        return err, code
    remove_index_record(ann_index)
    db.session.commit()
    return jsonify({'message': '索引已删除'})


# ──────────────────────────────────────────────
# kNN 搜索
# ──────────────────────────────────────────────

@index_bp.route('/indexes/<int:index_id>/search', methods=['POST'])
def search(index_id):
    db.session.expire_all()
    ann_index = AnnIndex.query.get_or_404(index_id)
    _, err, code = _can_access_index(ann_index)
    if err:
        return err, code

    if not index_is_usable(ann_index):
        status = effective_index_status(ann_index)
        return jsonify({'error': f'索引不可用 (状态: {status})，请重新构建'}), 400

    params_dict = json.loads(ann_index.params or '{}')

    # ── 联合索引走独立处理路径（7.3）──────────────────────────────
    if params_dict.get('joint'):
        return _joint_search(ann_index, params_dict)

    # ── 普通单数据集搜索 ──────────────────────────────────────────
    body       = request.get_json() or {}
    query_type = body.get('query_type', 'cell_id')
    top_k      = max(1, min(int(body.get('top_k', 10)), 200))
    nprobe     = max(1, int(body.get('nprobe', 10)))
    dataset_id = ann_index.dataset_id

    if dataset_id not in _dataset_cache:
        dataset = Dataset.query.get(dataset_id)
        if dataset is None:
            return jsonify({'error': '数据集不存在'}), 404
        data = load_dataset(dataset.file_path)
        cache_dataset(dataset_id, data)

    cache    = _dataset_cache[dataset_id]
    vectors  = cache['vectors']
    cell_ids = cache['cell_ids']
    obs      = cache['obs']

    t0 = time.time()

    if query_type == 'cell_id':
        cell_id = body.get('query_input')
        if cell_id is None:
            return jsonify({'error': '缺少 query_input（cell_id 字符串）'}), 400
        if cell_id not in cell_ids:
            return jsonify({'error': f'未找到 cell_id: {cell_id}'}), 404
        pos        = cell_ids.index(cell_id)
        query_vec  = vectors[pos]
        query_repr = cell_id

    elif query_type == 'vector':
        raw = body.get('query_input')
        if raw is None:
            return jsonify({'error': '缺少 query_input（浮点数列表）'}), 400
        query_vec = np.asarray(raw, dtype=np.float32)
        if query_vec.ndim != 1 or query_vec.shape[0] != vectors.shape[1]:
            return jsonify({
                'error': f'query_input 维度应为 {vectors.shape[1]}，'
                         f'实际为 {query_vec.shape}'
            }), 400
        query_repr = 'custom_vector'

    elif query_type == 'random':
        rng        = np.random.default_rng()
        pos        = int(rng.integers(0, len(cell_ids)))
        query_vec  = vectors[pos]
        query_repr = cell_ids[pos]

    else:
        return jsonify({'error': f'不支持的 query_type: {query_type}'}), 400

    try:
        indices, distances = search_index(ann_index, query_vec, k=top_k, nprobe=nprobe)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500

    elapsed_ms = round((time.time() - t0) * 1000, 2)

    results = []
    for rank, (pos, dist) in enumerate(zip(indices, distances), start=1):
        if not (0 <= pos < len(cell_ids)):
            continue
        row  = obs.iloc[pos]
        meta = {col: str(row[col]) for col in _META_COLS if col in obs.columns}
        results.append({'rank': rank, 'cell_id': cell_ids[pos],
                        'distance': float(dist), **meta})

    user_id = current_user.id if current_user.is_authenticated else None
    _write_history(user_id, dataset_id, query_type,
                   str(query_repr), ann_index.index_type, top_k,
                   [r['cell_id'] for r in results], elapsed_ms)

    return jsonify({
        'index_id':      index_id,
        'dataset_id':    dataset_id,
        'query_type':    query_type,
        'query_input':   str(query_repr),
        'top_k':         top_k,
        'query_time_ms': elapsed_ms,
        'results':       results,
    })


def _joint_search(ann_index: AnnIndex, params_dict: dict):
    """联合索引搜索处理（7.3）。"""
    body        = request.get_json() or {}
    query_type  = body.get('query_type', 'random')
    top_k       = max(1, min(int(body.get('top_k', 10)), 200))
    nprobe      = max(1, int(body.get('nprobe', 10)))
    dataset_ids = params_dict.get('dataset_ids', [])

    # 确保所有数据集均已缓存
    for ds_id in dataset_ids:
        if ds_id not in _dataset_cache:
            ds = Dataset.query.get(ds_id)
            if ds:
                data = load_dataset(ds.file_path)
                cache_dataset(ds_id, data)

    # 加载映射表（用于随机/cell_id 查询）
    import json as _json
    mapping_path = params_dict.get('mapping_file')
    mapping: list[dict] = []
    if mapping_path and os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = _json.load(f)

    t0 = time.time()
    exclude_cell_id = None
    query_cell_id   = None

    if query_type == 'vector':
        raw = body.get('query_input')
        if raw is None:
            return jsonify({'error': '缺少 query_input（浮点数列表）'}), 400
        query_vec  = np.asarray(raw, dtype=np.float32)
        query_repr = 'custom_vector'

    elif query_type == 'random':
        if not mapping:
            return jsonify({'error': '映射文件不存在，无法随机采样'}), 500
        rng   = np.random.default_rng()
        entry = mapping[int(rng.integers(0, len(mapping)))]
        ds_id = entry['dataset_id']
        pos   = entry['pos_in_dataset']
        query_vec  = _dataset_cache[ds_id]['vectors'][pos]
        query_repr = f"{ds_id}:{entry['cell_id']}"
        exclude_cell_id = entry['cell_id']
        query_cell_id   = entry['cell_id']

    elif query_type == 'cell_id':
        cell_id = body.get('query_input')
        if cell_id is None:
            return jsonify({'error': '缺少 query_input（cell_id 字符串）'}), 400
        found = False
        for ds_id in dataset_ids:
            if ds_id not in _dataset_cache:
                continue
            c = _dataset_cache[ds_id]
            if cell_id in c['cell_ids']:
                pos = c['cell_ids'].index(cell_id)
                query_vec  = c['vectors'][pos]
                query_repr = f'{ds_id}:{cell_id}'
                exclude_cell_id = cell_id
                query_cell_id   = cell_id
                found = True
                break
        if not found:
            return jsonify({'error': f'所有数据集中均未找到 cell_id: {cell_id}'}), 404

    else:
        return jsonify({'error': f'不支持的 query_type: {query_type}'}), 400

    search_k = top_k
    if exclude_cell_id:
        search_k = min(top_k * 10, 2000)

    try:
        raw_results = search_joint_index(ann_index, query_vec, k=search_k, nprobe=nprobe)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500

    elapsed_ms = round((time.time() - t0) * 1000, 2)

    # 附加元数据（排除查询细胞自身，与单库检索一致）
    results = []
    for item in raw_results:
        if exclude_cell_id and item['cell_id'] == exclude_cell_id:
            continue
        ds_id = item['dataset_id']
        pos   = item['pos_in_dataset']
        meta  = {}
        if ds_id in _dataset_cache:
            obs = _dataset_cache[ds_id]['obs']
            if pos < len(obs):
                row = obs.iloc[pos]
                meta = {col: str(row[col]) for col in _META_COLS if col in obs.columns}
        results.append({
            'rank':       len(results) + 1,
            'dataset_id': ds_id,
            'cell_id':    item['cell_id'],
            'distance':   item['distance'],
            **meta,
        })
        if len(results) >= top_k:
            break

    user_id = current_user.id if current_user.is_authenticated else None
    _write_history(user_id, dataset_ids[0] if dataset_ids else None,
                   query_type, str(query_repr), ann_index.index_type, top_k,
                   [r['cell_id'] for r in results], elapsed_ms)

    return jsonify({
        'index_id':      ann_index.id,
        'is_joint':      True,
        'dataset_ids':   dataset_ids,
        'query_type':    query_type,
        'query_input':   str(query_repr),
        'query_cell_id': query_cell_id or ('Vector' if query_type == 'vector' else None),
        'top_k':         top_k,
        'query_time_ms': elapsed_ms,
        'results':       results,
    })


# ──────────────────────────────────────────────
# 查询历史
# ──────────────────────────────────────────────

@index_bp.route('/history/', methods=['GET'])
@login_required_api
def list_history():
    dataset_id = request.args.get('dataset_id', type=int)
    limit      = min(200, max(1, int(request.args.get('limit', 50))))

    q = QueryHistory.query.order_by(QueryHistory.created_at.desc())
    if dataset_id:
        q = q.filter_by(dataset_id=dataset_id)
    if current_user.role not in ('sysadmin', 'labadmin'):
        q = q.filter_by(user_id=current_user.id)

    records = q.limit(limit).all()
    return jsonify([_history_to_dict(h) for h in records])


# ──────────────────────────────────────────────
# 内部辅助函数
# ──────────────────────────────────────────────

def _write_history(user_id, dataset_id, query_type, query_input,
                   index_type, top_k, result_cell_ids, query_time_ms):
    """写入查询历史记录。"""
    history = QueryHistory(
        user_id    = user_id,
        dataset_id = dataset_id,
        query_type = query_type,
        query_input= query_input[:512],
        index_type = index_type,
        top_k      = top_k,
        result_ids = json.dumps(result_cell_ids, ensure_ascii=False),
        query_time = query_time_ms,
    )
    db.session.add(history)
    db.session.commit()


def _index_to_dict(i: AnnIndex) -> dict:
    return {
        'id':         i.id,
        'dataset_id': i.dataset_id,
        'index_type': i.index_type,
        'metric':     i.metric,
        'params':     json.loads(i.params) if i.params else {},
        'index_file': i.index_file,
        'status':     effective_index_status(i),
        'build_time': i.build_time,
        'created_at': i.created_at.isoformat(),
    }


def _history_to_dict(h: QueryHistory) -> dict:
    return {
        'id':          h.id,
        'user_id':     h.user_id,
        'dataset_id':  h.dataset_id,
        'query_type':  h.query_type,
        'query_input': h.query_input,
        'index_type':  h.index_type,
        'top_k':       h.top_k,
        'query_time':  h.query_time,
        'result_ids':  json.loads(h.result_ids) if h.result_ids else [],
        'created_at':  h.created_at.isoformat(),
    }
