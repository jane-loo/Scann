import json
import os

import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, current_app
from flask_login import current_user
from werkzeug.utils import secure_filename

from ..models import db, Dataset, AnnIndex
from ..decorators import login_required_api
from ..permissions import (
    visible_datasets_query, get_accessible_dataset, can_manage_data,
)
from .loader import (
    validate_h5ad, load_dataset, cache_dataset,
    get_cached_dataset, evict_cache, _dataset_cache,
)
from ..index.manager import effective_index_status

data_bp = Blueprint('data', __name__)

# obs 中优先返回给前端的列（其余列也保留，放在后面）
_PRIORITY_COLS = ['cell_type', 'disease', 'AgeGroup', 'donor_id',
                  'sex', 'tissue', 'author_cell_type', 'Phase']


# ──────────────────────────────────────────────
# 上传数据集
# ──────────────────────────────────────────────

@data_bp.route('/upload', methods=['POST'])
@login_required_api
def upload_dataset():
    if not can_manage_data():
        return jsonify({'error': '访客无权上传数据，请先登录'}), 403
    if 'file' not in request.files:
        return jsonify({'error': '未选择文件'}), 400

    file = request.files['file']
    if not file or not file.filename.lower().endswith('.h5ad'):
        return jsonify({'error': '只支持 .h5ad 格式'}), 400

    name        = request.form.get('name', file.filename).strip()
    description = request.form.get('description', '').strip()

    # 安全文件名，并在重名时自动加序号
    filename  = secure_filename(file.filename)
    save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    base, ext = os.path.splitext(save_path)
    counter   = 1
    while os.path.exists(save_path):
        save_path = f'{base}_{counter}{ext}'
        counter  += 1

    file.save(save_path)

    # 校验文件合法性
    valid, err = validate_h5ad(save_path)
    if not valid:
        os.remove(save_path)
        return jsonify({'error': err}), 400

    # 读取元数据
    data = load_dataset(save_path)

    dataset = Dataset(
        name        = name,
        description = description,
        file_path   = save_path,
        n_cells     = data['n_cells'],
        n_genes     = data['n_genes'],
        n_dims      = data['n_dims'],
        cell_types  = json.dumps(data['cell_types'],  ensure_ascii=False),
        obs_columns = json.dumps(data['obs_columns'], ensure_ascii=False),
        upload_by   = current_user.id,
    )
    db.session.add(dataset)
    db.session.commit()

    # 放入内存缓存
    cache_dataset(dataset.id, data)

    return jsonify({
        'message': '上传成功',
        'dataset': _dataset_to_dict(dataset),
    }), 201


# ──────────────────────────────────────────────
# 数据集列表
# ──────────────────────────────────────────────

@data_bp.route('/', methods=['GET'])
def list_datasets():
    datasets = visible_datasets_query().order_by(Dataset.created_at.desc()).all()
    return jsonify([_dataset_to_dict(d) for d in datasets])


# ──────────────────────────────────────────────
# 数据集详情
# ──────────────────────────────────────────────

@data_bp.route('/<int:dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    dataset, err, code = get_accessible_dataset(dataset_id)
    if err:
        return err, code
    data    = _ensure_cached(dataset_id, dataset.file_path)

    # 计算所有列的唯一值分布
    all_filters = {}
    obs = data['obs']
    for col in obs.columns:
        # 只为分类变量（非连续数值且唯一值不多的）生成下拉列表
        if obs[col].dtype == 'object' or obs[col].dtype.name == 'category':
            # 限制唯一值数量，避免下拉框太长
            unique_vals = obs[col].dropna().unique().tolist()
            if len(unique_vals) <= 100:
                all_filters[col] = unique_vals

    result = _dataset_to_dict(dataset)
    result['cell_type_distribution'] = all_filters  # 重用该字段传输所有过滤项
    result['indexes'] = [_index_to_dict(i) for i in dataset.indexes]
    return jsonify(result)


# ──────────────────────────────────────────────
# 删除数据集
# ──────────────────────────────────────────────

@data_bp.route('/<int:dataset_id>', methods=['DELETE'])
@login_required_api
def delete_dataset(dataset_id):
    dataset, err, code = get_accessible_dataset(dataset_id)
    if err:
        return err, code
    from ..permissions import effective_role
    if effective_role() not in ('sysadmin', 'labadmin') and dataset.upload_by != current_user.id:
        return jsonify({'error': '无权删除该数据集'}), 403

    # 删除关联索引文件
    for idx in dataset.indexes:
        if idx.index_file and os.path.exists(idx.index_file):
            os.remove(idx.index_file)

    # 删除 .h5ad 文件
    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)

    # 清除内存缓存
    evict_cache(dataset_id)

    db.session.delete(dataset)
    db.session.commit()
    return jsonify({'message': '删除成功'})


# ──────────────────────────────────────────────
# 细胞列表（分页）
# ──────────────────────────────────────────────

@data_bp.route('/<int:dataset_id>/cells', methods=['GET'])
def list_cells(dataset_id):
    dataset, err, code = get_accessible_dataset(dataset_id)
    if err:
        return err, code
    page     = max(1, int(request.args.get('page', 1)))
    per_page = min(200, max(1, int(request.args.get('per_page', 50))))

    data     = _ensure_cached(dataset_id, dataset.file_path)
    obs      = data['obs']
    cell_ids = data['cell_ids']
    total    = len(cell_ids)

    start = (page - 1) * per_page
    end   = min(start + per_page, total)

    # 选取优先列（存在的部分）+ 其余列
    priority = [c for c in _PRIORITY_COLS if c in obs.columns]
    others   = [c for c in obs.columns if c not in priority]
    ordered  = priority + others

    cells = []
    for i in range(start, end):
        row  = obs.iloc[i]
        meta = {col: _safe_val(row[col]) for col in ordered}
        cells.append({'cell_id': cell_ids[i], **meta})

    return jsonify({
        'total':    total,
        'page':     page,
        'per_page': per_page,
        'pages':    (total + per_page - 1) // per_page,
        'cells':    cells,
    })


# ──────────────────────────────────────────────
# 两细胞对比（查询 vs 命中）
# ──────────────────────────────────────────────

@data_bp.route('/<int:dataset_id>/cells/compare', methods=['POST'])
def compare_cells(dataset_id):
    """返回查询细胞与命中细胞的元数据、向量维度对比及嵌入坐标。"""
    dataset, err, code = get_accessible_dataset(dataset_id)
    if err:
        return err, code

    body = request.get_json() or {}
    query_cell_id  = (body.get('query_cell_id') or '').strip()
    target_cell_id = (body.get('target_cell_id') or '').strip()
    search_distance  = body.get('search_distance')
    search_similarity = body.get('search_similarity')

    if not target_cell_id:
        return jsonify({'error': '缺少 target_cell_id'}), 400

    data     = _ensure_cached(dataset_id, dataset.file_path)
    obs      = data['obs']
    vectors  = data['vectors']
    cell_ids = data['cell_ids']

    try:
        ti = cell_ids.index(target_cell_id)
    except ValueError:
        return jsonify({'error': f'未找到细胞 {target_cell_id}'}), 404

    target_snap = _cell_snapshot(cell_ids, obs, ti)
    query_snap  = None
    qi          = None

    if query_cell_id and query_cell_id not in ('Vector', ''):
        try:
            qi = cell_ids.index(query_cell_id)
            query_snap = _cell_snapshot(cell_ids, obs, qi)
        except ValueError:
            pass

    metadata_rows = []
    if query_snap:
        for col in obs.columns:
            qv = query_snap['metadata'].get(col)
            tv = target_snap['metadata'].get(col)
            metadata_rows.append({
                'field': col,
                'query_value': qv,
                'target_value': tv,
                'match': qv == tv,
            })

    preview_dims = min(15, vectors.shape[1])
    vector_dims  = []
    metrics      = {
        'n_dims_total': int(vectors.shape[1]),
        'distance':     search_distance,
        'similarity':   search_similarity,
    }

    if query_snap:
        qv = vectors[qi]
        tv = vectors[ti]
        dist = float(np.linalg.norm(qv - tv))
        metrics['distance']   = dist
        metrics['similarity'] = round(1.0 / (1.0 + dist), 6)
        for d in range(preview_dims):
            vector_dims.append({
                'dim':    d + 1,
                'query':  float(qv[d]),
                'target': float(tv[d]),
                'delta':  float(abs(qv[d] - tv[d])),
            })
    else:
        tv = vectors[ti]
        for d in range(preview_dims):
            vector_dims.append({
                'dim': d + 1, 'query': None,
                'target': float(tv[d]), 'delta': None,
            })

    embed_2d = {'target': _embed_2d(data, vectors, ti)}
    if qi is not None:
        embed_2d['query'] = _embed_2d(data, vectors, qi)

    return jsonify({
        'query':         query_snap,
        'target':        target_snap,
        'metadata_rows': metadata_rows,
        'vector_dims':   vector_dims,
        'metrics':       metrics,
        'embed_2d':      embed_2d,
        'has_query':     query_snap is not None,
    })


def _cell_snapshot(cell_ids, obs, idx: int) -> dict:
    row = obs.iloc[idx]
    return {
        'cell_id':  cell_ids[idx],
        'metadata': {col: _safe_val(row[col]) for col in obs.columns},
        'cell_type': _safe_val(row['cell_type']) if 'cell_type' in obs.columns else None,
    }


def _embed_2d(data: dict, vectors, idx: int) -> dict:
    out = {
        'pca_x': float(vectors[idx, 0]),
        'pca_y': float(vectors[idx, 1]),
    }
    umap = data.get('umap_coords')
    if umap is not None:
        out['umap_x'] = float(umap[idx, 0])
        out['umap_y'] = float(umap[idx, 1])
    return out


# ──────────────────────────────────────────────
# UMAP 坐标（供可视化）
# ──────────────────────────────────────────────

@data_bp.route('/<int:dataset_id>/umap_data', methods=['GET'])
def get_umap_data(dataset_id):
    dataset, err, code = get_accessible_dataset(dataset_id)
    if err:
        return err, code
    data    = _ensure_cached(dataset_id, dataset.file_path)

    if data['umap_coords'] is None:
        return jsonify({'error': '该数据集无 UMAP 坐标'}), 404

    obs      = data['obs']
    coords   = data['umap_coords']
    cell_ids = data['cell_ids']

    result = {
        'cell_ids':   cell_ids,
        'umap_x':     coords[:, 0].tolist(),
        'umap_y':     coords[:, 1].tolist(),
        'cell_types': (obs['cell_type'].astype(str).tolist()
                       if 'cell_type' in obs.columns
                       else ['unknown'] * len(cell_ids)),
    }
    # 附加过滤字段
    for col in ['disease', 'AgeGroup', 'donor_id', 'sex']:
        if col in obs.columns:
            result[col] = obs[col].astype(str).tolist()

    return jsonify(result)


# ──────────────────────────────────────────────
# PCA 坐标（前两维，供可视化备用）
# ──────────────────────────────────────────────

@data_bp.route('/<int:dataset_id>/pca_data', methods=['GET'])
def get_pca_data(dataset_id):
    dataset, err, code = get_accessible_dataset(dataset_id)
    if err:
        return err, code
    data    = _ensure_cached(dataset_id, dataset.file_path)

    vectors  = data['vectors']
    obs      = data['obs']
    cell_ids = data['cell_ids']

    return jsonify({
        'cell_ids':   cell_ids,
        'pca_x':      vectors[:, 0].tolist(),
        'pca_y':      vectors[:, 1].tolist(),
        'cell_types': (obs['cell_type'].astype(str).tolist()
                       if 'cell_type' in obs.columns
                       else ['unknown'] * len(cell_ids)),
    })


# ──────────────────────────────────────────────
# 内部辅助函数
# ──────────────────────────────────────────────

def _ensure_cached(dataset_id: int, file_path: str) -> dict:
    """确保数据集在内存缓存中，否则加载。"""
    if dataset_id not in _dataset_cache:
        data = load_dataset(file_path)
        cache_dataset(dataset_id, data)
    return _dataset_cache[dataset_id]


def _safe_val(v):
    """将 pandas/numpy 值转为可 JSON 序列化的 Python 原生类型。"""
    if isinstance(v, float) and np.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _dataset_to_dict(d: Dataset) -> dict:
    return {
        'id':          d.id,
        'name':        d.name,
        'description': d.description,
        'n_cells':     d.n_cells,
        'n_genes':     d.n_genes,
        'n_dims':      d.n_dims,
        'cell_types':  json.loads(d.cell_types)  if d.cell_types  else [],
        'obs_columns': json.loads(d.obs_columns) if d.obs_columns else [],
        'created_at':  d.created_at.isoformat(),
    }


def _index_to_dict(i: AnnIndex) -> dict:
    return {
        'id':         i.id,
        'index_type': i.index_type,
        'metric':     i.metric,
        'status':     effective_index_status(i),
        'build_time': i.build_time,
        'created_at': i.created_at.isoformat(),
    }
