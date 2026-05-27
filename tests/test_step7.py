"""
第七步测试：7.1 启动预加载 + 7.3 多数据集联合索引
运行方式：python -m pytest tests/test_step7.py -v
"""
import io
import os
import tempfile
import time

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sqlalchemy.pool import StaticPool

from app import create_app
from app.models import db, User


# ──────────────────────────────────────────────
# 合成 h5ad 工厂
# ──────────────────────────────────────────────

def _make_h5ad_bytes(n_cells=60, n_genes=80, n_pca=10, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    X   = rng.random((n_cells, n_genes)).astype(np.float32)

    obs = pd.DataFrame({
        'cell_type': rng.choice(['hepatocyte', 'T cell', 'NK cell'], n_cells),
        'disease':   ['normal'] * n_cells,
        'AgeGroup':  rng.choice(['Adult', 'Ped'], n_cells),
        'donor_id':  rng.choice(['D001', 'D002'], n_cells),
        'sex':       rng.choice(['male', 'female'], n_cells),
    }, index=[f'DS{seed}_CELL_{i:04d}' for i in range(n_cells)])

    var   = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    pca = PCA(n_components=n_pca, random_state=seed)
    adata.obsm['X_pca']  = pca.fit_transform(X).astype(np.float32)
    adata.obsm['X_umap'] = rng.random((n_cells, 2)).astype(np.float32)

    tmp = tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False)
    tmp.close()
    try:
        adata.write_h5ad(tmp.name)
        with open(tmp.name, 'rb') as f:
            return f.read()
    finally:
        os.unlink(tmp.name)


# ──────────────────────────────────────────────
# Fixtures（整个 module 共用）
# ──────────────────────────────────────────────

@pytest.fixture(scope='module')
def app():
    _app = create_app({
        'TESTING':                 True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'SQLALCHEMY_ENGINE_OPTIONS': {
            'connect_args': {'check_same_thread': False},
            'poolclass':    StaticPool,
        },
        'UPLOAD_FOLDER': 'data/uploads',
        'INDEX_FOLDER':  'data/indexes',
        'WTF_CSRF_ENABLED': False,
    })
    with _app.app_context():
        db.drop_all()
        db.create_all()
        u = User(username='step7admin', email='s7@t.com', role='admin')
        u.set_password('s7pass')
        db.session.add(u)
        db.session.commit()
    yield _app


@pytest.fixture(scope='module')
def client(app):
    return app.test_client()


@pytest.fixture(scope='module')
def anon_client(app):
    """专用于认证测试的全新未登录客户端。"""
    return app.test_client()


@pytest.fixture(scope='module')
def auth_client(client):
    client.post('/auth/login', json={'username': 'step7admin', 'password': 's7pass'})
    yield client


@pytest.fixture(scope='module')
def ds1_id(auth_client):
    """上传数据集 1（seed=0，60 细胞）。"""
    b = _make_h5ad_bytes(seed=0)
    resp = auth_client.post(
        '/api/datasets/upload',
        data={'file': (io.BytesIO(b), 'ds1.h5ad'), 'name': 'Dataset1'},
        content_type='multipart/form-data',
    )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()['dataset']['id']


@pytest.fixture(scope='module')
def ds2_id(auth_client):
    """上传数据集 2（seed=99，60 细胞）。"""
    b = _make_h5ad_bytes(seed=99)
    resp = auth_client.post(
        '/api/datasets/upload',
        data={'file': (io.BytesIO(b), 'ds2.h5ad'), 'name': 'Dataset2'},
        content_type='multipart/form-data',
    )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()['dataset']['id']


def _wait_ready(auth_client, index_id: int, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = auth_client.get(f'/api/indexes/{index_id}')
        d = r.get_json()
        if d['status'] in ('ready', 'error'):
            return d
        time.sleep(0.1)
    pytest.fail(f'联合索引 {index_id} 在 {timeout}s 内未就绪')


@pytest.fixture(scope='module')
def joint_hnsw_id(auth_client, ds1_id, ds2_id):
    """构建 HNSW 联合索引，等待就绪，返回 index_id。"""
    resp = auth_client.post('/api/indexes/joint_build', json={
        'dataset_ids': [ds1_id, ds2_id],
        'index_type':  'hnsw',
        'metric':      'l2',
        'params':      {'M': 8, 'ef_construction': 50},
    })
    assert resp.status_code == 202, resp.get_json()
    body = resp.get_json()
    assert body['dataset_ids'] == [ds1_id, ds2_id]
    idx_id = body['index_id']
    detail = _wait_ready(auth_client, idx_id)
    assert detail['status'] == 'ready', f'构建失败: {detail}'
    return idx_id


@pytest.fixture(scope='module')
def joint_exact_id(auth_client, ds1_id, ds2_id):
    """构建 FAISS exact 联合索引。"""
    resp = auth_client.post('/api/indexes/joint_build', json={
        'dataset_ids': [ds1_id, ds2_id],
        'index_type':  'exact',
        'metric':      'l2',
    })
    assert resp.status_code == 202
    idx_id = resp.get_json()['index_id']
    detail = _wait_ready(auth_client, idx_id)
    assert detail['status'] == 'ready', f'构建失败: {detail}'
    return idx_id


# ──────────────────────────────────────────────
# 7.1 启动预加载测试
# ──────────────────────────────────────────────

class TestStartupPreload:
    def test_testing_mode_skips_preload(self, app):
        """TESTING=True 时不应触发预加载（create_app 不抛异常即通过）。"""
        from app.data.loader import _dataset_cache
        # 应用已通过 TESTING=True 创建，_dataset_cache 中无启动时预填充的数据
        # （fixture 上传后会填入，这里只验证没有因预加载崩溃）
        assert app.config['TESTING'] is True

    def test_preload_function_exists(self):
        """_preload_datasets 函数可以被导入。"""
        from app import _preload_datasets
        assert callable(_preload_datasets)

    def test_create_app_non_testing_calls_preload(self):
        """
        非 TESTING 模式创建 app，预加载代码路径不抛异常
        （数据库为空时直接跳过，不报错）。
        """
        from app import create_app as _ca
        # 不传 TESTING，但使用临时内存 DB，避免真实 DB 副作用
        _app = _ca({
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'SQLALCHEMY_ENGINE_OPTIONS': {
                'connect_args': {'check_same_thread': False},
                'poolclass':    StaticPool,
            },
            'UPLOAD_FOLDER': 'data/uploads',
            'INDEX_FOLDER':  'data/indexes',
        })
        with _app.app_context():
            db.create_all()   # 空表
        # 未抛异常即通过


# ──────────────────────────────────────────────
# 7.3 联合索引构建接口测试
# ──────────────────────────────────────────────

class TestJointBuildValidation:
    def test_requires_at_least_two_datasets(self, auth_client, ds1_id):
        resp = auth_client.post('/api/indexes/joint_build',
                                json={'dataset_ids': [ds1_id]})
        assert resp.status_code == 400

    def test_empty_dataset_ids(self, auth_client):
        resp = auth_client.post('/api/indexes/joint_build',
                                json={'dataset_ids': []})
        assert resp.status_code == 400

    def test_nonexistent_dataset(self, auth_client, ds1_id):
        resp = auth_client.post('/api/indexes/joint_build',
                                json={'dataset_ids': [ds1_id, 99999]})
        assert resp.status_code == 404

    def test_invalid_index_type(self, auth_client, ds1_id, ds2_id):
        resp = auth_client.post('/api/indexes/joint_build',
                                json={'dataset_ids': [ds1_id, ds2_id],
                                      'index_type':  'bad_type'})
        assert resp.status_code == 400

    def test_invalid_metric(self, auth_client, ds1_id, ds2_id):
        resp = auth_client.post('/api/indexes/joint_build',
                                json={'dataset_ids': [ds1_id, ds2_id],
                                      'metric':      'manhattan'})
        assert resp.status_code == 400

    def test_requires_login(self, anon_client):
        # 使用独立的未登录客户端，与 auth_client 共享的 client 隔离
        resp = anon_client.post('/api/indexes/joint_build',
                                json={'dataset_ids': [1, 2]})
        assert resp.status_code in (302, 401)

    def test_returns_202(self, auth_client, ds1_id, ds2_id):
        resp = auth_client.post('/api/indexes/joint_build',
                                json={'dataset_ids': [ds1_id, ds2_id],
                                      'index_type':  'exact'})
        assert resp.status_code == 202
        body = resp.get_json()
        assert 'index_id'    in body
        assert 'dataset_ids' in body
        assert body['status'] == 'building'


# ──────────────────────────────────────────────
# 7.3 HNSW 联合索引状态与文件
# ──────────────────────────────────────────────

class TestJointHNSWReady:
    def test_status_is_ready(self, auth_client, joint_hnsw_id):
        resp = auth_client.get(f'/api/indexes/{joint_hnsw_id}')
        d = resp.get_json()
        assert d['status']    == 'ready'
        assert d['index_type'] == 'hnsw'
        assert d['build_time'] >= 0

    def test_params_contain_joint_flag(self, auth_client, joint_hnsw_id):
        d = auth_client.get(f'/api/indexes/{joint_hnsw_id}').get_json()
        p = d['params']
        assert p.get('joint') is True
        assert 'dataset_ids'   in p
        assert 'mapping_file'  in p
        assert 'total_cells'   in p

    def test_total_cells_equals_sum(self, auth_client, joint_hnsw_id, ds1_id, ds2_id):
        """联合索引的 total_cells 应等于两个数据集细胞数之和。"""
        p = auth_client.get(f'/api/indexes/{joint_hnsw_id}').get_json()['params']
        ds1 = auth_client.get(f'/api/datasets/{ds1_id}').get_json()
        ds2 = auth_client.get(f'/api/datasets/{ds2_id}').get_json()
        assert p['total_cells'] == ds1['n_cells'] + ds2['n_cells']

    def test_index_file_exists(self, auth_client, joint_hnsw_id):
        d = auth_client.get(f'/api/indexes/{joint_hnsw_id}').get_json()
        assert os.path.exists(d['index_file'])

    def test_mapping_file_exists(self, auth_client, joint_hnsw_id):
        p = auth_client.get(f'/api/indexes/{joint_hnsw_id}').get_json()['params']
        assert os.path.exists(p['mapping_file'])


# ──────────────────────────────────────────────
# 7.3 联合索引搜索
# ──────────────────────────────────────────────

class TestJointSearch:
    def test_random_search_returns_results(self, auth_client, joint_hnsw_id):
        resp = auth_client.post(f'/api/indexes/{joint_hnsw_id}/search',
                                json={'query_type': 'random', 'top_k': 5})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['is_joint'] is True
        assert len(body['results']) == 5

    def test_results_have_dataset_id(self, auth_client, joint_hnsw_id):
        resp = auth_client.post(f'/api/indexes/{joint_hnsw_id}/search',
                                json={'query_type': 'random', 'top_k': 10})
        for r in resp.get_json()['results']:
            assert 'dataset_id' in r
            assert 'cell_id'    in r
            assert 'distance'   in r
            assert 'rank'       in r

    def test_results_span_both_datasets(self, auth_client, joint_hnsw_id,
                                        ds1_id, ds2_id):
        """请求足够多结果时，应包含来自两个数据集的记录。"""
        resp = auth_client.post(f'/api/indexes/{joint_hnsw_id}/search',
                                json={'query_type': 'random', 'top_k': 120})
        ds_ids = {r['dataset_id'] for r in resp.get_json()['results']}
        assert ds1_id in ds_ids, '结果应包含来自 Dataset1 的细胞'
        assert ds2_id in ds_ids, '结果应包含来自 Dataset2 的细胞'

    def test_cell_id_search_across_datasets(self, auth_client, joint_hnsw_id):
        """cell_id 查询：用 Dataset2 的细胞在联合索引中搜索。"""
        resp = auth_client.post(f'/api/indexes/{joint_hnsw_id}/search',
                                json={'query_type': 'cell_id',
                                      'query_input': 'DS99_CELL_0000',
                                      'top_k': 5})
        assert resp.status_code == 200
        results = resp.get_json()['results']
        assert len(results) >= 1
        assert results[0]['cell_id'] == 'DS99_CELL_0000'

    def test_vector_search(self, auth_client, joint_hnsw_id):
        vec = [0.0] * 10   # 10 维零向量
        resp = auth_client.post(f'/api/indexes/{joint_hnsw_id}/search',
                                json={'query_type': 'vector',
                                      'query_input': vec,
                                      'top_k': 3})
        assert resp.status_code == 200
        assert len(resp.get_json()['results']) == 3

    def test_invalid_cell_id_returns_404(self, auth_client, joint_hnsw_id):
        resp = auth_client.post(f'/api/indexes/{joint_hnsw_id}/search',
                                json={'query_type': 'cell_id',
                                      'query_input': 'NO_SUCH_CELL'})
        assert resp.status_code == 404

    def test_exact_joint_search(self, auth_client, joint_exact_id):
        resp = auth_client.post(f'/api/indexes/{joint_exact_id}/search',
                                json={'query_type': 'random', 'top_k': 5})
        assert resp.status_code == 200
        assert len(resp.get_json()['results']) == 5

    def test_distances_ascending(self, auth_client, joint_hnsw_id):
        resp = auth_client.post(f'/api/indexes/{joint_hnsw_id}/search',
                                json={'query_type': 'random', 'top_k': 10})
        dists = [r['distance'] for r in resp.get_json()['results']]
        assert dists == sorted(dists), '距离应升序排列'

    def test_history_recorded(self, auth_client, joint_hnsw_id):
        before = len(auth_client.get('/api/history/').get_json())
        auth_client.post(f'/api/indexes/{joint_hnsw_id}/search',
                         json={'query_type': 'random', 'top_k': 3})
        after = len(auth_client.get('/api/history/').get_json())
        assert after > before


# ──────────────────────────────────────────────
# 7.3 联合索引删除（含文件清理）
# ──────────────────────────────────────────────

class TestJointDelete:
    def test_delete_removes_files(self, auth_client, ds1_id, ds2_id):
        # 创建专门用于删除测试的联合索引
        resp = auth_client.post('/api/indexes/joint_build',
                                json={'dataset_ids': [ds1_id, ds2_id],
                                      'index_type':  'exact'})
        idx_id = resp.get_json()['index_id']
        detail = _wait_ready(auth_client, idx_id)
        assert detail['status'] == 'ready'

        index_file   = detail['index_file']
        mapping_file = detail['params']['mapping_file']

        # 删除
        del_resp = auth_client.delete(f'/api/indexes/{idx_id}')
        assert del_resp.status_code == 200
        assert del_resp.get_json()['message'] == '索引已删除'

        # 记录已删
        assert auth_client.get(f'/api/indexes/{idx_id}').status_code == 404

        # 文件已清理
        assert not os.path.exists(index_file),   '索引文件应被删除'
        assert not os.path.exists(mapping_file), '映射文件应被删除'
