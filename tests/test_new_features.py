import pytest
import io
import json
import time
import numpy as np
import pandas as pd
import anndata as ad
import tempfile
import os
from app import create_app
from app.models import db, User, Dataset, AnnIndex, EvaluationReport, QueryHistory
from sqlalchemy.pool import StaticPool

def _make_h5ad(n_cells=50, n_genes=30):
    X = np.random.random((n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame({
        'cell_type': np.random.choice(['A', 'B', 'C'], n_cells),
        'disease': ['healthy'] * n_cells
    }, index=[f'cell_{i}' for i in range(n_cells)])
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm['X_pca'] = X # 简化测试
    
    tmp = tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False)
    tmp.close()
    adata.write_h5ad(tmp.name)
    return tmp.name

@pytest.fixture
def client():
    # 使用临时文件数据库，避免 :memory: 在多线程下的 sqlite3 (DatabaseError)
    db_fd, db_path = tempfile.mkstemp()
    
    app = create_app({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{db_path}',
        'UPLOAD_FOLDER': tempfile.gettempdir(),
        'INDEX_FOLDER': tempfile.gettempdir(),
        'WTF_CSRF_ENABLED': False,
    })
    
    with app.app_context():
        db.create_all()
        # 创建一个普通用户和一个管理员
        u1 = User(username='user', email='u@t.com', role='user')
        u1.set_password('pass123')
        u2 = User(username='admin', email='a@t.com', role='admin')
        u2.set_password('admin123')
        db.session.add_all([u1, u2])
        db.session.commit()
        
    yield app.test_client()
    
    # 清理数据库文件
    os.close(db_fd)
    if os.path.exists(db_path):
        os.remove(db_path)

def login(client, username, password):
    return client.post('/auth/login', json={'username': username, 'password': password})

def test_full_workflow(client):
    # 1. 登录
    login(client, 'admin', 'admin123')
    
    # 2. 上传数据集
    h5ad_path = _make_h5ad()
    with open(h5ad_path, 'rb') as f:
        res = client.post('/api/datasets/upload', data={
            'file': (f, 'test.h5ad'),
            'name': 'Test Dataset'
        })
    assert res.status_code == 201
    ds_id = res.get_json()['dataset']['id']
    
    # 3. 构建索引 (Exact 和 HNSW)
    # 构建 Exact 作为 GT
    res = client.post('/api/indexes/build', json={
        'dataset_id': ds_id,
        'index_type': 'exact',
        'metric': 'l2'
    })
    assert res.status_code == 202
    
    # 构建 HNSW
    res = client.post('/api/indexes/build', json={
        'dataset_id': ds_id,
        'index_type': 'hnsw',
        'metric': 'l2',
        'params': {'M': 16, 'ef_construction': 100}
    })
    assert res.status_code == 202
    
    # 等待索引构建完成 (由于是测试模式下的异步线程，可能需要一点时间)
    # 在这个特殊的测试配置下，build_index_sync 会在后台运行
    # 简单起见，我们轮询一下状态
    timeout = 10
    start = time.time()
    while time.time() - start < timeout:
        res = client.get(f'/api/indexes/?dataset_id={ds_id}')
        indices = res.get_json()
        if len(indices) >= 2 and all(idx['status'] == 'ready' for idx in indices):
            break
        time.sleep(1)
    
    indices = client.get(f'/api/indexes/?dataset_id={ds_id}').get_json()
    hnsw_idx_id = next(i['id'] for i in indices if i['index_type'] == 'hnsw')
    
    # 4. 测试搜索模块 (Search Module)
    # By Cell ID
    res = client.post('/api/search/by_cell_id', json={
        'dataset_id': ds_id,
        'index_id': hnsw_idx_id,
        'cell_id': 'cell_0',
        'top_k': 5
    })
    assert res.status_code == 200
    data = res.get_json()
    assert 'results' in data
    assert len(data['results']) == 5
    
    # Random
    res = client.post('/api/search/random', json={
        'dataset_id': ds_id,
        'index_id': hnsw_idx_id,
        'top_k': 3
    })
    assert res.status_code == 200
    
    # 5. 测试评测模块 (Evaluation Module)
    res = client.post(f'/api/evaluate/{ds_id}', json={
        'index_id': hnsw_idx_id,
        'k': 10,
        'n_queries': 5
    })
    assert res.status_code == 200
    eval_data = res.get_json()
    assert 'report_id' in eval_data
    assert eval_data['results']['recall_at_k'] >= 0
    
    # 获取报告
    res = client.get(f'/api/evaluate/{ds_id}/report')
    assert res.status_code == 200
    report = res.get_json()
    assert report['index_id'] == hnsw_idx_id

    # 6. 测试管理员模块 (Admin Module)
    # 获取用户列表
    res = client.get('/admin/users')
    assert res.status_code == 200
    users = res.get_json()
    assert len(users) >= 2
    
    # 修改用户
    user_id = next(u['id'] for u in users if u['username'] == 'user')
    res = client.put(f'/admin/users/{user_id}', json={'role': 'admin'})
    assert res.status_code == 200
    
    # 检查历史记录
    res = client.get('/api/history/')
    assert res.status_code == 200
    history = res.get_json()
    assert len(history) > 0 # 之前的搜索应该有记录

    print("\n[V] 所有新功能验证通过！")

if __name__ == "__main__":
    # 这里只是为了方便演示，实际建议用 pytest 运行
    pass
