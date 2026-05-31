"""
数据集管理 API 测试
运行方式：python -m pytest tests/test_data_api.py -v
"""
import io
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.decomposition import PCA

from sqlalchemy.pool import StaticPool

from app import create_app
from app.models import db, User


# ──────────────────────────────────────────────
# 测试 h5ad 工厂（写到临时文件再读回 bytes）
# anndata 0.10.x 不支持直接写入 BytesIO，需要真实路径
# ──────────────────────────────────────────────

def _make_h5ad_bytes(n_cells=50, n_genes=100, n_pca=10) -> bytes:
    rng = np.random.default_rng(42)
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

    pca = PCA(n_components=n_pca, random_state=42)
    adata.obsm['X_pca']  = pca.fit_transform(X).astype(np.float32)
    adata.obsm['X_umap'] = rng.random((n_cells, 2)).astype(np.float32)

    # anndata 0.10.x 必须写到真实路径
    tmp = tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False)
    tmp.close()
    try:
        adata.write_h5ad(tmp.name)
        with open(tmp.name, 'rb') as f:
            return f.read()
    finally:
        os.unlink(tmp.name)


# ──────────────────────────────────────────────
# Fixtures（全部 module 级，共用同一个 app / db / 登录状态）
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
        'UPLOAD_FOLDER':           'data/uploads',
        'INDEX_FOLDER':            'data/indexes',
        'WTF_CSRF_ENABLED':        False,
    })
    with _app.app_context():
        db.drop_all()
        db.create_all()
        u = User(username='testadmin', email='admin@t.com', role='admin')
        u.set_password('test123')
        db.session.add(u)
        db.session.commit()
    yield _app


@pytest.fixture(scope='module')
def client(app):
    return app.test_client()


@pytest.fixture(scope='module')
def auth_client(client):
    """已登录的测试客户端（整个模块共用，避免重复登录）。"""
    client.post('/auth/login',
                json={'username': 'testadmin', 'password': 'test123'})
    yield client


@pytest.fixture(scope='module')
def h5ad_bytes():
    """合成 h5ad 字节，module 级只生成一次。"""
    return _make_h5ad_bytes()


@pytest.fixture(scope='module')
def uploaded_dataset_id(auth_client, h5ad_bytes):
    """上传一个测试数据集，返回其 id，整个模块共用。"""
    resp = auth_client.post(
        '/api/datasets/upload',
        data={
            'file':        (io.BytesIO(h5ad_bytes), 'test_liver.h5ad'),
            'name':        'Test Liver',
            'description': '单元测试数据集',
        },
        content_type='multipart/form-data',
    )
    assert resp.status_code == 201, f'上传失败: {resp.get_json()}'
    return resp.get_json()['dataset']['id']


# ──────────────────────────────────────────────
# 认证保护
# ──────────────────────────────────────────────

class TestAuth:
    def test_list_datasets_requires_login(self, client):
        resp = client.get('/api/datasets/')
        assert resp.status_code in (302, 401)

    def test_login_success(self, client):
        resp = client.post('/auth/login',
                           json={'username': 'testadmin', 'password': 'test123'})
        assert resp.status_code == 200
        assert resp.get_json()['message'] == '登录成功'

    def test_login_wrong_password(self, client):
        resp = client.post('/auth/login',
                           json={'username': 'testadmin', 'password': 'wrong'})
        assert resp.status_code == 401


# ──────────────────────────────────────────────
# 上传数据集
# ──────────────────────────────────────────────

class TestUploadDataset:
    def test_upload_valid_h5ad(self, auth_client, h5ad_bytes):
        resp = auth_client.post(
            '/api/datasets/upload',
            data={
                'file':        (io.BytesIO(h5ad_bytes), 'test_liver.h5ad'),
                'name':        'Test Liver',
                'description': '单元测试数据集',
            },
            content_type='multipart/form-data',
        )
        assert resp.status_code == 201, resp.get_json()
        ds = resp.get_json()['dataset']
        assert ds['name']    == 'Test Liver'
        assert ds['n_cells'] == 50
        assert ds['n_genes'] == 100
        assert ds['n_dims']  == 10
        assert 'hepatocyte' in ds['cell_types']
        assert 'cell_type'  in ds['obs_columns']
        assert 'disease'    in ds['obs_columns']
        assert 'AgeGroup'   in ds['obs_columns']

    def test_upload_wrong_extension(self, auth_client):
        resp = auth_client.post(
            '/api/datasets/upload',
            data={'file': (io.BytesIO(b'hello'), 'bad.txt')},
            content_type='multipart/form-data',
        )
        assert resp.status_code == 400
        assert '.h5ad' in resp.get_json()['error']

    def test_upload_no_file(self, auth_client):
        resp = auth_client.post('/api/datasets/upload', data={},
                                content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_list_after_upload(self, auth_client, uploaded_dataset_id):
        resp = auth_client.get('/api/datasets/')
        assert resp.status_code == 200
        ids = [d['id'] for d in resp.get_json()]
        assert uploaded_dataset_id in ids


# ──────────────────────────────────────────────
# 数据集详情
# ──────────────────────────────────────────────

class TestDatasetDetail:
    def test_get_detail(self, auth_client, uploaded_dataset_id):
        resp = auth_client.get(f'/api/datasets/{uploaded_dataset_id}')
        assert resp.status_code == 200
        d = resp.get_json()
        assert d['n_cells'] == 50
        assert 'cell_type_distribution' in d
        assert 'indexes'                in d
        ct_dist = d['cell_type_distribution']
        assert len(ct_dist) >= 1

    def test_get_nonexistent(self, auth_client):
        resp = auth_client.get('/api/datasets/99999')
        assert resp.status_code == 404


# ──────────────────────────────────────────────
# 细胞列表（分页）
# ──────────────────────────────────────────────

class TestCellList:
    def test_default_page(self, auth_client, uploaded_dataset_id):
        resp = auth_client.get(f'/api/datasets/{uploaded_dataset_id}/cells')
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['total']    == 50
        assert body['page']     == 1
        assert body['per_page'] == 50
        assert len(body['cells']) == 50

    def test_pagination(self, auth_client, uploaded_dataset_id):
        resp = auth_client.get(
            f'/api/datasets/{uploaded_dataset_id}/cells?page=1&per_page=20')
        body = resp.get_json()
        assert len(body['cells']) == 20
        assert body['pages']      == 3   # ceil(50/20)

    def test_page_2(self, auth_client, uploaded_dataset_id):
        resp = auth_client.get(
            f'/api/datasets/{uploaded_dataset_id}/cells?page=2&per_page=20')
        assert len(resp.get_json()['cells']) == 20

    def test_page_3_partial(self, auth_client, uploaded_dataset_id):
        resp = auth_client.get(
            f'/api/datasets/{uploaded_dataset_id}/cells?page=3&per_page=20')
        assert len(resp.get_json()['cells']) == 10   # 50 - 40

    def test_cell_has_key_fields(self, auth_client, uploaded_dataset_id):
        resp = auth_client.get(
            f'/api/datasets/{uploaded_dataset_id}/cells?per_page=1')
        cell = resp.get_json()['cells'][0]
        assert 'cell_id'   in cell
        assert 'cell_type' in cell
        assert 'disease'   in cell
        assert 'AgeGroup'  in cell
        assert cell['cell_id'].startswith('CELL_')

    def test_cell_values_serializable(self, auth_client, uploaded_dataset_id):
        """所有字段值必须是 JSON 原生类型（无 NaN、numpy 类型等）。"""
        import json
        resp  = auth_client.get(
            f'/api/datasets/{uploaded_dataset_id}/cells?per_page=10')
        # 能序列化即通过
        json.dumps(resp.get_json())


# ──────────────────────────────────────────────
# UMAP 坐标
# ──────────────────────────────────────────────

class TestUmapData:
    def test_umap_fields(self, auth_client, uploaded_dataset_id):
        resp = auth_client.get(
            f'/api/datasets/{uploaded_dataset_id}/umap_data')
        assert resp.status_code == 200
        d = resp.get_json()
        for key in ('cell_ids', 'umap_x', 'umap_y', 'cell_types'):
            assert key in d, f'缺少字段: {key}'

    def test_umap_length(self, auth_client, uploaded_dataset_id):
        resp = auth_client.get(
            f'/api/datasets/{uploaded_dataset_id}/umap_data')
        d    = resp.get_json()
        n    = len(d['cell_ids'])
        assert n == 50
        assert len(d['umap_x'])     == n
        assert len(d['umap_y'])     == n
        assert len(d['cell_types']) == n

    def test_umap_extra_cols(self, auth_client, uploaded_dataset_id):
        resp = auth_client.get(
            f'/api/datasets/{uploaded_dataset_id}/umap_data')
        d    = resp.get_json()
        assert 'disease'  in d
        assert 'AgeGroup' in d


# ──────────────────────────────────────────────
# PCA 坐标
# ──────────────────────────────────────────────

class TestPcaData:
    def test_pca_data(self, auth_client, uploaded_dataset_id):
        resp = auth_client.get(
            f'/api/datasets/{uploaded_dataset_id}/pca_data')
        assert resp.status_code == 200
        d = resp.get_json()
        assert len(d['pca_x']) == 50
        assert len(d['pca_y']) == 50
        assert len(d['cell_types']) == 50


# ──────────────────────────────────────────────
# 删除数据集
# ──────────────────────────────────────────────

class TestDeleteDataset:
    def test_delete(self, auth_client, h5ad_bytes):
        # 上传专用于删除的数据集
        up = auth_client.post(
            '/api/datasets/upload',
            data={'file': (io.BytesIO(h5ad_bytes), 'to_delete.h5ad'),
                  'name': 'ToDelete'},
            content_type='multipart/form-data',
        )
        ds_id = up.get_json()['dataset']['id']

        resp = auth_client.delete(f'/api/datasets/{ds_id}')
        assert resp.status_code == 200
        assert resp.get_json()['message'] == '删除成功'

        # 删除后应返回 404
        assert auth_client.get(f'/api/datasets/{ds_id}').status_code == 404

    def test_delete_nonexistent(self, auth_client):
        resp = auth_client.delete('/api/datasets/99999')
        assert resp.status_code == 404
