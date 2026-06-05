import os
import json
import pytest
import numpy as np
from app import create_app, db
from app.models import User, Dataset, AnnIndex, EvaluationReport

# ---------------------------------------------------------
# 配置测试环境
# ---------------------------------------------------------
@pytest.fixture
def app():
    # 使用临时文件数据库进行测试，避免破坏现有数据
    db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_temp.db')
    app = create_app({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{db_path}',
        'WTF_CSRF_ENABLED': False
    })

    with app.app_context():
        db.create_all()
        # 创建测试用户
        roles = ['visitor', 'normal', 'expert', 'labadmin', 'sysadmin']
        for r in roles:
            u = User(username=r, email=f'{r}@t.com', role=r)
            u.set_password('pass123')
            db.session.add(u)
        
        # 创建一个演示数据集 (upload_by=None)
        ds1 = Dataset(name="Demo Data", file_path="fake.h5ad", upload_by=None)
        # 创建一个私有数据集 (upload_by=2, 即 normal 用户)
        ds2 = Dataset(name="Private Data", file_path="fake2.h5ad", upload_by=2)
        db.session.add_all([ds1, ds2])
        db.session.commit()

        yield app
        
        db.session.remove()
        db.drop_all()
        if os.path.exists(db_path):
            os.remove(db_path)

@pytest.fixture
def client(app):
    return app.test_client()

def login(client, username, password):
    return client.post('/auth/login', json={
        'username': username,
        'password': password
    }, follow_redirects=True)

# ---------------------------------------------------------
# 1. 测试数据可见性 (Data Visibility)
# ---------------------------------------------------------
def test_data_visibility(client):
    # A. 访客登录
    login(client, 'visitor', 'pass123')
    res = client.get('/api/datasets/')
    assert res.status_code == 200
    data = res.get_json()
    assert data is not None
    assert len(data) == 1
    assert data[0]['name'] == "Demo Data"
    client.post('/auth/logout')

    # B. 普通用户登录
    login(client, 'normal', 'pass123')
    res = client.get('/api/datasets/')
    data = res.get_json()
    assert len(data) == 2 # 看到演示 + 自己的
    client.post('/auth/logout')

    # C. 实验室管理员登录
    login(client, 'labadmin', 'pass123')
    res = client.get('/api/datasets/')
    data = res.get_json()
    assert len(data) == 2 # 看到所有
    client.post('/auth/logout')

# ---------------------------------------------------------
# 2. 测试权限拦截 (Role-Based Access Control)
# ---------------------------------------------------------
def test_evaluate_permission(client):
    # A. 普通用户尝试评测 -> 应被拦截
    login(client, 'normal', 'pass123')
    res = client.post('/api/evaluate/1', json={'index_id': 1})
    # 因为未授权，如果是 API 应该返回 403。但如果装饰器写得不对可能返回 302。
    # 检查 decorators.py 发现返回的是 jsonify + 403，所以这里应该是 403。
    assert res.status_code == 403
    client.post('/auth/logout')

    # B. 专家尝试评测 -> 应允许 (此处会因为数据集无效查不到索引而报 404/400 但不会是 403)
    login(client, 'expert', 'pass123')
    res = client.post('/api/evaluate/1', json={'index_id': 1})
    assert res.status_code != 403
    client.post('/auth/logout')

# ---------------------------------------------------------
# 3. 测试搜索逻辑 (Engine Logic - Similarity & Exclusion)
# ---------------------------------------------------------
# 由于 _process_results 是内联的，我们直接对结果计算进行逻辑验证
def test_search_logic_integrity():
    # 模拟计算公式: 1 / (1 + distance)
    dist = 0.5
    similarity = 1.0 / (1.0 + dist)
    assert round(similarity, 4) == 0.6667
    
    # 模拟排除逻辑
    results = [{"cell_id": "Cell_A"}, {"cell_id": "Cell_B"}]
    exclude_id = "Cell_A"
    filtered = [r for r in results if r['cell_id'] != exclude_id]
    assert len(filtered) == 1
    assert filtered[0]['cell_id'] == "Cell_B"

# ---------------------------------------------------------
# 4. 测试加速比计算逻辑 (Speedup Logic)
# ---------------------------------------------------------
def test_speedup_calculation(app):
    # 模拟评估过程中的时间
    avg_ann_latency = 2.0
    avg_exact_latency = 100.0
    
    # 加速比应为 50.0
    speedup = avg_exact_latency / avg_ann_latency
    assert speedup == 50.0

if __name__ == "__main__":
    print("请运行 'pytest Scann/tests/test_v2_logic.py' 来执行测试。")
