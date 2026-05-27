"""
ANN 索引构建与搜索测试
运行方式：python -m pytest tests/test_index.py -v
"""
import io
import os
import tempfile
import time

import numpy as np
import pandas as pd
import anndata as ad
import pytest
from sklearn.decomposition import PCA
from sqlalchemy.pool import StaticPool

from app import create_app
from app.models import db, User


# ──────────────────────────────────────────────
# 合成 h5ad 文件工厂（同 test_data_api.py）
# ──────────────────────────────────────────────

def _make_h5ad_bytes(n_cells=80, n_genes=100, n_pca=10) -> bytes:
    rng = np.random.default_rng(0)
    X   = rng.random((n_cells, n_genes)).astype(np.float32)

    obs = pd.DataFrame({
        'cell_type': rng.choice(['hepatocyte', 'T cell', 'NK cell'], n_cells),
        'disease':   ['normal'] * n_cells,
        'AgeGroup':  rng.choice(['Adult', 'Ped'], n_cells),
        'donor_id':  rng.choice(['D001', 'D002'], n_cells),
        'sex':       rng.choice(['male', 'female'], n_cells),
    }, index=[f'CELL_{i:04d}' for i in range(n_cells)])

    var   = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    pca = PCA(n_components=n_pca, random_state=0)
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
# Fixtures
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
        u = User(username='idxadmin', email='idx@t.com', role='admin')
        u.set_password('idx123')
        db.session.add(u)
        db.session.commit()
    yield _app


@pytest.fixture(scope='module')
def client(app):
    return app.test_client()


@pytest.fixture(scope='module')
def auth_client(client):
    client.post('/auth/login',
                json={'username': 'idxadmin', 'password': 'idx123'})
    yield client


@pytest.fixture(scope='module')
def h5ad_bytes():
    return _make_h5ad_bytes()


@pytest.fixture(scope='module')
def dataset_id(auth_client, h5ad_bytes):
    """上传测试数据集，返回 dataset_id。"""
    resp = auth_client.post(
        '/api/datasets/upload',
        data={
            'file':        (io.BytesIO(h5ad_bytes), 'idx_test.h5ad'),
            'name':        'IndexTest',
            'description': '索引测试数据集',
        },
        content_type='multipart/form-data',
    )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()['dataset']['id']


def _wait_ready(auth_client, index_id: int, timeout: float = 15.0) -> dict:
    """轮询直到索引状态变为 ready 或 error，返回最终详情。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = auth_client.get(f'/api/indexes/{index_id}')
        d = r.get_json()
        if d['status'] in ('ready', 'error'):
            return d
        time.sleep(0.1)
    pytest.fail(f'索引 {index_id} 在 {timeout}s 内未变为 ready')


@pytest.fixture(scope='module')
def hnsw_index_id(auth_client, dataset_id):
    """构建 HNSW l2 索引，等待 ready，返回 index_id。"""
    resp = auth_client.post('/api/indexes/build', json={
        'dataset_id': dataset_id,
        'index_type': 'hnsw',
        'metric':     'l2',
        'params':     {'M': 8, 'ef_construction': 50},
    })
    assert resp.status_code == 202, resp.get_json()
    idx_id = resp.get_json()['index_id']
    detail = _wait_ready(auth_client, idx_id)
    assert detail['status'] == 'ready', f'构建失败: {detail}'
    return idx_id


@pytest.fixture(scope='module')
def faiss_exact_index_id(auth_client, dataset_id):
    """构建 FAISS exact l2 索引，等待 ready，返回 index_id。"""
    resp = auth_client.post('/api/indexes/build', json={
        'dataset_id': dataset_id,
        'index_type': 'exact',
        'metric':     'l2',
    })
    assert resp.status_code == 202, resp.get_json()
    idx_id = resp.get_json()['index_id']
    detail = _wait_ready(auth_client, idx_id)
    assert detail['status'] == 'ready', f'构建失败: {detail}'
    return idx_id


@pytest.fixture(scope='module')
def faiss_ivf_index_id(auth_client, dataset_id):
    """构建 FAISS ivf_flat cosine 索引，等待 ready，返回 index_id。"""
    resp = auth_client.post('/api/indexes/build', json={
        'dataset_id': dataset_id,
        'index_type': 'ivf_flat',
        'metric':     'cosine',
        'params':     {'nlist': 8},
    })
    assert resp.status_code == 202, resp.get_json()
    idx_id = resp.get_json()['index_id']
    detail = _wait_ready(auth_client, idx_id)
    assert detail['status'] == 'ready', f'构建失败: {detail}'
    return idx_id


# ──────────────────────────────────────────────
# 认证保护
# ──────────────────────────────────────────────

class TestIndexAuth:
    def test_build_requires_login(self, client):
        # 不使用 dataset_id fixture（避免触发 auth_client 初始化）
        resp = client.post('/api/indexes/build',
                           json={'dataset_id': 99999})
        assert resp.status_code in (302, 401)

    def test_list_requires_login(self, client):
        resp = client.get('/api/indexes/')
        assert resp.status_code in (302, 401)


# ──────────────────────────────────────────────
# 构建接口
# ──────────────────────────────────────────────

class TestBuildIndex:
    def test_build_returns_202(self, auth_client, dataset_id):
        resp = auth_client.post('/api/indexes/build', json={
            'dataset_id': dataset_id,
            'index_type': 'hnsw',
            'metric':     'l2',
        })
        assert resp.status_code == 202
        body = resp.get_json()
        assert 'index_id' in body
        assert body['status'] == 'building'

    def test_build_invalid_index_type(self, auth_client, dataset_id):
        resp = auth_client.post('/api/indexes/build', json={
            'dataset_id': dataset_id,
            'index_type': 'unknown_type',
        })
        assert resp.status_code == 400

    def test_build_invalid_metric(self, auth_client, dataset_id):
        resp = auth_client.post('/api/indexes/build', json={
            'dataset_id': dataset_id,
            'index_type': 'hnsw',
            'metric':     'manhattan',
        })
        assert resp.status_code == 400

    def test_build_missing_dataset(self, auth_client):
        resp = auth_client.post('/api/indexes/build', json={})
        assert resp.status_code == 400

    def test_build_nonexistent_dataset(self, auth_client):
        resp = auth_client.post('/api/indexes/build', json={
            'dataset_id': 99999,
            'index_type': 'hnsw',
        })
        assert resp.status_code == 404


# ──────────────────────────────────────────────
# HNSW 索引
# ──────────────────────────────────────────────

class TestHNSWIndex:
    def test_status_is_ready(self, auth_client, hnsw_index_id):
        resp = auth_client.get(f'/api/indexes/{hnsw_index_id}')
        assert resp.status_code == 200
        d = resp.get_json()
        assert d['status']     == 'ready'
        assert d['index_type'] == 'hnsw'
        assert d['metric']     == 'l2'
        assert d['build_time'] >= 0

    def test_index_file_exists(self, auth_client, hnsw_index_id):
        resp = auth_client.get(f'/api/indexes/{hnsw_index_id}')
        path = resp.get_json()['index_file']
        assert path and os.path.exists(path)

    def test_search_by_cell_id(self, auth_client, hnsw_index_id):
        resp = auth_client.post(
            f'/api/indexes/{hnsw_index_id}/search',
            json={'query_type': 'cell_id', 'query_input': 'CELL_0000', 'top_k': 5},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert len(body['results']) == 5
        assert body['results'][0]['cell_id'] == 'CELL_0000'   # 第一个是自身

    def test_search_by_random(self, auth_client, hnsw_index_id):
        resp = auth_client.post(
            f'/api/indexes/{hnsw_index_id}/search',
            json={'query_type': 'random', 'top_k': 3},
        )
        assert resp.status_code == 200
        assert len(resp.get_json()['results']) == 3

    def test_search_result_has_metadata(self, auth_client, hnsw_index_id):
        resp = auth_client.post(
            f'/api/indexes/{hnsw_index_id}/search',
            json={'query_type': 'cell_id', 'query_input': 'CELL_0001', 'top_k': 1},
        )
        result = resp.get_json()['results'][0]
        assert 'cell_id'   in result
        assert 'distance'  in result
        assert 'rank'      in result
        assert 'cell_type' in result

    def test_search_result_distances_ascending(self, auth_client, hnsw_index_id):
        """距离应按升序排列。"""
        resp = auth_client.post(
            f'/api/indexes/{hnsw_index_id}/search',
            json={'query_type': 'cell_id', 'query_input': 'CELL_0005', 'top_k': 10},
        )
        dists = [r['distance'] for r in resp.get_json()['results']]
        assert dists == sorted(dists), '距离应升序排列'

    def test_search_top_k_capped_at_200(self, auth_client, hnsw_index_id):
        resp = auth_client.post(
            f'/api/indexes/{hnsw_index_id}/search',
            json={'query_type': 'random', 'top_k': 9999},
        )
        assert len(resp.get_json()['results']) <= 200

    def test_search_invalid_cell_id(self, auth_client, hnsw_index_id):
        resp = auth_client.post(
            f'/api/indexes/{hnsw_index_id}/search',
            json={'query_type': 'cell_id', 'query_input': 'NO_SUCH_CELL'},
        )
        assert resp.status_code == 404

    def test_search_by_vector(self, auth_client, hnsw_index_id, dataset_id):
        # 拿一个真实向量测试
        cells_resp = auth_client.get(
            f'/api/datasets/{dataset_id}/pca_data')
        pca_data = cells_resp.get_json()
        vec = [pca_data['pca_x'][0]] + [pca_data['pca_y'][0]] + [0.0] * 8

        resp = auth_client.post(
            f'/api/indexes/{hnsw_index_id}/search',
            json={'query_type': 'vector', 'query_input': vec, 'top_k': 3},
        )
        assert resp.status_code == 200
        assert len(resp.get_json()['results']) == 3

    def test_search_vector_wrong_dim(self, auth_client, hnsw_index_id):
        resp = auth_client.post(
            f'/api/indexes/{hnsw_index_id}/search',
            json={'query_type': 'vector', 'query_input': [0.1, 0.2]},
        )
        assert resp.status_code == 400


# ──────────────────────────────────────────────
# FAISS 精确索引
# ──────────────────────────────────────────────

class TestFaissExactIndex:
    def test_status_is_ready(self, auth_client, faiss_exact_index_id):
        resp = auth_client.get(f'/api/indexes/{faiss_exact_index_id}')
        d = resp.get_json()
        assert d['status']     == 'ready'
        assert d['index_type'] == 'exact'

    def test_search_returns_results(self, auth_client, faiss_exact_index_id):
        resp = auth_client.post(
            f'/api/indexes/{faiss_exact_index_id}/search',
            json={'query_type': 'random', 'top_k': 5},
        )
        assert resp.status_code == 200
        assert len(resp.get_json()['results']) == 5

    def test_search_self_is_first(self, auth_client, faiss_exact_index_id):
        """精确索引：用某细胞自身向量查询，第一个结果应是自身。"""
        resp = auth_client.post(
            f'/api/indexes/{faiss_exact_index_id}/search',
            json={'query_type': 'cell_id', 'query_input': 'CELL_0010', 'top_k': 5},
        )
        assert resp.get_json()['results'][0]['cell_id'] == 'CELL_0010'


# ──────────────────────────────────────────────
# FAISS IVF 索引
# ──────────────────────────────────────────────

class TestFaissIVFIndex:
    def test_status_is_ready(self, auth_client, faiss_ivf_index_id):
        resp = auth_client.get(f'/api/indexes/{faiss_ivf_index_id}')
        d = resp.get_json()
        assert d['status']     == 'ready'
        assert d['index_type'] == 'ivf_flat'
        assert d['metric']     == 'cosine'

    def test_search_returns_results(self, auth_client, faiss_ivf_index_id):
        resp = auth_client.post(
            f'/api/indexes/{faiss_ivf_index_id}/search',
            json={'query_type': 'random', 'top_k': 5},
        )
        assert resp.status_code == 200
        assert len(resp.get_json()['results']) == 5


# ──────────────────────────────────────────────
# 索引列表
# ──────────────────────────────────────────────

class TestListIndexes:
    def test_list_all(self, auth_client):
        resp = auth_client.get('/api/indexes/')
        assert resp.status_code == 200
        assert isinstance(resp.get_json(), list)
        assert len(resp.get_json()) >= 3   # hnsw + exact + ivf_flat + 测试中建的

    def test_list_by_dataset(self, auth_client, dataset_id):
        resp = auth_client.get(f'/api/indexes/?dataset_id={dataset_id}')
        assert resp.status_code == 200
        for item in resp.get_json():
            assert item['dataset_id'] == dataset_id

    def test_list_nonexistent_dataset(self, auth_client):
        resp = auth_client.get('/api/indexes/?dataset_id=99999')
        assert resp.status_code == 200
        assert resp.get_json() == []


# ──────────────────────────────────────────────
# 搜索未就绪索引
# ──────────────────────────────────────────────

class TestSearchNotReady:
    def test_search_building_index_returns_400(self, auth_client, dataset_id):
        """对 building 状态的索引搜索，应返回 400。"""
        # 先建一个，马上搜（状态为 building）
        resp = auth_client.post('/api/indexes/build', json={
            'dataset_id': dataset_id,
            'index_type': 'hnsw',
            'metric':     'l2',
        })
        idx_id = resp.get_json()['index_id']
        # 立刻搜索，状态大概率还是 building
        # 如果碰巧已经 ready 则跳过（小数据集有可能瞬间完成）
        detail = auth_client.get(f'/api/indexes/{idx_id}').get_json()
        if detail['status'] == 'building':
            r = auth_client.post(f'/api/indexes/{idx_id}/search',
                                 json={'query_type': 'random'})
            assert r.status_code == 400


# ──────────────────────────────────────────────
# 删除索引
# ──────────────────────────────────────────────

class TestDeleteIndex:
    def test_delete(self, auth_client, dataset_id, h5ad_bytes):
        # 构建一个专用于删除的索引
        resp = auth_client.post('/api/indexes/build', json={
            'dataset_id': dataset_id,
            'index_type': 'exact',
            'metric':     'l2',
        })
        idx_id = resp.get_json()['index_id']
        detail = _wait_ready(auth_client, idx_id)
        assert detail['status'] == 'ready'
        index_file = detail['index_file']

        # 删除
        del_resp = auth_client.delete(f'/api/indexes/{idx_id}')
        assert del_resp.status_code == 200
        assert del_resp.get_json()['message'] == '索引已删除'

        # 已删除 → 404
        assert auth_client.get(f'/api/indexes/{idx_id}').status_code == 404

        # 文件已清理
        if index_file:
            assert not os.path.exists(index_file)

    def test_delete_nonexistent(self, auth_client):
        resp = auth_client.delete('/api/indexes/99999')
        assert resp.status_code == 404


# ──────────────────────────────────────────────
# 查询历史
# ──────────────────────────────────────────────

class TestQueryHistory:
    def test_history_recorded(self, auth_client, hnsw_index_id):
        """执行一次搜索后，历史记录数量应增加。"""
        before = len(auth_client.get('/api/history/').get_json())
        auth_client.post(
            f'/api/indexes/{hnsw_index_id}/search',
            json={'query_type': 'random', 'top_k': 3},
        )
        after = len(auth_client.get('/api/history/').get_json())
        assert after > before

    def test_history_by_dataset(self, auth_client, dataset_id):
        resp = auth_client.get(f'/api/history/?dataset_id={dataset_id}')
        assert resp.status_code == 200
        assert isinstance(resp.get_json(), list)
