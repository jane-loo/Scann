"""
无 X_pca 上传时自动 PCA 向量化测试
运行：python -m pytest tests/test_auto_pca.py -v
"""
import io
import os
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from sqlalchemy.pool import StaticPool

from app import create_app
from app.data.loader import (
    SCANN_UNS_KEY,
    load_dataset,
    ensure_pca_vectors,
)
from app.models import db, User


def _make_raw_h5ad_path(n_cells=40, n_genes=120, seed=7) -> str:
    rng = np.random.default_rng(seed)
    X = (rng.random((n_cells, n_genes)) * 50).astype(np.float32)
    obs = pd.DataFrame(
        {'cell_type': rng.choice(['A', 'B'], n_cells)},
        index=[f'Cell_{i:03d}' for i in range(n_cells)],
    )
    adata = ad.AnnData(X=X, obs=obs)
    tmp = tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False)
    tmp.close()
    adata.write_h5ad(tmp.name)
    return tmp.name


@pytest.fixture(scope='module')
def app():
    _app = create_app({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'SQLALCHEMY_ENGINE_OPTIONS': {
            'connect_args': {'check_same_thread': False},
            'poolclass': StaticPool,
        },
        'UPLOAD_FOLDER': 'data/uploads',
        'INDEX_FOLDER': 'data/indexes',
        'WTF_CSRF_ENABLED': False,
    })
    with _app.app_context():
        db.drop_all()
        db.create_all()
        u = User(username='pcauser', email='pca@t.com', role='admin')
        u.set_password('test123')
        db.session.add(u)
        db.session.commit()
    yield _app


@pytest.fixture(scope='module')
def auth_client(app):
    client = app.test_client()
    client.post('/auth/login', json={'username': 'pcauser', 'password': 'test123'})
    return client


class TestAutoPcaLoader:
    def test_load_dataset_computes_pca_when_missing(self):
        path = _make_raw_h5ad_path()
        try:
            data = load_dataset(path, persist_pca=True)
            assert data['pca_computed'] is True
            assert data['vectorization']['source'] == 'auto_pca'
            assert data['vectors'].shape == (40, 30)
            assert data['n_dims'] == 30

            reread = sc.read_h5ad(path)
            assert 'X_pca' in reread.obsm
            assert SCANN_UNS_KEY in reread.uns
            assert reread.uns[SCANN_UNS_KEY]['computed_by'] == 'Scann 上传向量化流水线'
        finally:
            os.unlink(path)

    def test_second_load_uses_file_pca(self):
        path = _make_raw_h5ad_path(seed=99)
        try:
            first = load_dataset(path, persist_pca=True)
            second = load_dataset(path, persist_pca=True)
            assert first['pca_computed'] is True
            assert second['pca_computed'] is False
            assert second['vectorization']['source'] == 'auto_pca'
            np.testing.assert_array_equal(first['vectors'], second['vectors'])
        finally:
            os.unlink(path)


class TestAutoPcaUpload:
    def test_upload_without_x_pca(self, auth_client):
        path = _make_raw_h5ad_path(seed=11)
        try:
            with open(path, 'rb') as f:
                resp = auth_client.post(
                    '/api/datasets/upload',
                    data={
                        'file': (io.BytesIO(f.read()), 'raw_counts.h5ad'),
                        'name': 'Raw Upload',
                    },
                    content_type='multipart/form-data',
                )
            assert resp.status_code == 201, resp.get_json()
            body = resp.get_json()
            assert body['pca_computed'] is True
            assert body['vectorization']['source'] == 'auto_pca'
            assert body['dataset']['vector_source'] == 'auto_pca'
            assert body['dataset']['n_dims'] == 30
        finally:
            os.unlink(path)
