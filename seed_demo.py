import os
import json
import numpy as np
import pandas as pd
import anndata as ad
from app import create_app, db
from app.models import User, Dataset, AnnIndex
from app.data.loader import cache_dataset

def create_sample_h5ad(path):
    n_cells = 100
    n_genes = 200
    n_dims = 30
    
    # 随机生成 X
    X = np.random.rand(n_cells, n_genes).astype(np.float32)
    # 模拟降维结果
    X_pca = np.random.rand(n_cells, n_dims).astype(np.float32)
    
    # 元数据
    obs = pd.DataFrame({
        'cell_type': np.random.choice(['T cell', 'B cell', 'Monocyte'], n_cells),
        'disease': np.random.choice(['Normal', 'COVID-19'], n_cells),
        'AgeGroup': np.random.choice(['Adult', 'Elderly'], n_cells),
        'donor_id': np.random.choice(['D1', 'D2', 'D3'], n_cells)
    }, index=[f'Cell_{i:03d}' for i in range(n_cells)])
    
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm['X_pca'] = X_pca
    adata.write_h5ad(path)
    return n_cells, n_genes, n_dims

app = create_app()
with app.app_context():
    # 确保用户存在
    admin = User.query.filter_by(username='sysadmin').first()
    if not admin:
        admin = User(username='sysadmin', email='admin@scann.com', role='sysadmin')
        admin.set_password('pass123')
        db.session.add(admin)
        db.session.commit()

    # 创建上传目录
    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # 1. 生成示例数据集 1
    path1 = os.path.join(upload_dir, 'sample_peripheral_blood.h5ad')
    n_c, n_g, n_d = create_sample_h5ad(path1)
    
    ds1 = Dataset(
        name='外周血单细胞对照组',
        description='这是一个包含100个细胞的演示数据集，用于测试检索功能。',
        file_path=path1,
        n_cells=n_c,
        n_genes=n_g,
        n_dims=n_d,
        cell_types=json.dumps(['T cell', 'B cell', 'Monocyte']),
        obs_columns=json.dumps(['cell_type', 'disease', 'AgeGroup', 'donor_id']),
        upload_by=admin.id
    )
    db.session.add(ds1)
    db.session.commit()
    print(f"创建数据集: {ds1.name}")

    # 2. 为该数据集创建一个 HNSW 索引 (标记为 ready)
    idx1 = AnnIndex(
        dataset_id=ds1.id,
        index_type='hnsw',
        metric='l2',
        params=json.dumps({'dim': n_d, 'M': 16, 'efConstruction': 200}),
        status='ready'
    )
    db.session.add(idx1)
    db.session.commit()
    print(f"创建索引: {idx1.index_type} for {ds1.name}")

    # 3. 生成示例数据集 2 (公开数据)
    path2 = os.path.join(upload_dir, 'public_atlas.h5ad')
    n_c, n_g, n_d = create_sample_h5ad(path2)
    ds2 = Dataset(
        name='人类肺部细胞图谱 (Demo)',
        description='公共演示数据，访客权限即可访问。',
        file_path=path2,
        n_cells=n_c,
        n_genes=n_g,
        n_dims=n_d,
        cell_types=json.dumps(['Epithelial', 'Endothelial', 'Stromal']),
        obs_columns=json.dumps(['cell_type', 'disease']),
        upload_by=None # 公开
    )
    db.session.add(ds2)
    db.session.commit()
    print(f"创建公开数据集: {ds2.name}")

    # 强制缓存数据集以加快首次访问
    data1 = {'n_cells': n_c, 'n_dims': n_d, 'cells': [f'Cell_{i:03d}' for i in range(100)], 'pca': np.random.rand(100, n_d).astype(np.float32)}
    cache_dataset(ds1.id, data1)
    
    print("数据初始化成功！")