import os
import json
import numpy as np
import pandas as pd
import anndata as ad
from app import create_app, db
from app.models import User, Dataset, AnnIndex

def create_sample_h5ad(path, name):
    n_cells, n_genes, n_dims = 100, 200, 30
    adata = ad.AnnData(
        X=np.random.rand(n_cells, n_genes).astype(np.float32),
        obs=pd.DataFrame({'cell_type': ['TypeA']*50+['TypeB']*50}, index=[f'Cell_{i:03d}' for i in range(100)])
    )
    adata.obsm['X_pca'] = np.random.rand(n_cells, n_dims).astype(np.float32)
    adata.write_h5ad(path)
    return n_cells, n_genes, n_dims

app = create_app()
with app.app_context():
    db.drop_all()
    db.create_all()
    
    admin = User(username='sysadmin', email='admin@scann.com', role='sysadmin')
    admin.set_password('pass123')
    db.session.add(admin)
    db.session.commit()
    
    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_dir): os.makedirs(upload_dir)
    
    datasets = [
        ('外周血演示数据', 'demo_blood.h5ad'),
        ('人肺部实验数据', 'demo_lung.h5ad')
    ]
    
    for name, fname in datasets:
        path = os.path.join(upload_dir, fname)
        c, g, d = create_sample_h5ad(path, name)
        
        # 提取细胞类型和列名（与 app/data/loader.py 逻辑一致）
        cell_types = ['TypeA', 'TypeB']
        obs_columns = ['cell_type']
        
        # 设置 upload_by=None 使其在所有角色（包括 visitor）下可见
        ds = Dataset(
            name=name, 
            file_path=path, 
            n_cells=c, 
            n_genes=g, 
            n_dims=d, 
            cell_types=json.dumps(cell_types),
            obs_columns=json.dumps(obs_columns),
            upload_by=None
        )
        db.session.add(ds)
        db.session.commit()
        
        # 为每个数据集预装索引
        db.session.add(AnnIndex(dataset_id=ds.id, index_type='hnsw', status='ready', metric='l2', params='{"dim":30}'))
        db.session.add(AnnIndex(dataset_id=ds.id, index_type='exact', status='ready', metric='l2', params='{"dim":30}'))
    
    db.session.commit()
    print("Database Reset Done with 2 Datasets.")