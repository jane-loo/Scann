import os
import json
import shutil
import numpy as np
import pandas as pd
import anndata as ad
from app import create_app, db
from app.models import User, Dataset, AnnIndex
from app.data.loader import load_dataset, cache_dataset
from app.index.manager import build_index_sync

def create_sample_h5ad(path, name):
    n_cells, n_genes, n_dims = 100, 200, 30
    adata = ad.AnnData(
        X=np.random.rand(n_cells, n_genes).astype(np.float32),
        obs=pd.DataFrame({'cell_type': ['TypeA']*50+['TypeB']*50}, index=[f'Cell_{i:03d}' for i in range(100)])
    )
    adata.obsm['X_pca'] = np.random.rand(n_cells, n_dims).astype(np.float32)
    # 演示用 2D 嵌入（PCA 前两维），供前端散点图展示
    adata.obsm['X_umap'] = adata.obsm['X_pca'][:, :2].astype(np.float32)
    adata.write_h5ad(path)
    return n_cells, n_genes, n_dims

def build_dataset_index(app, dataset_id, index_type, n_dims, index_folder):
    params = {'dim': n_dims}
    ann_index = AnnIndex(
        dataset_id=dataset_id,
        index_type=index_type,
        metric='l2',
        params=json.dumps(params),
        status='building',
    )
    db.session.add(ann_index)
    db.session.commit()
    ann_index_id = ann_index.id
    try:
        build_index_sync(
            app,
            dataset_id=dataset_id,
            ann_index_id=ann_index_id,
            index_type=index_type,
            metric='l2',
            params=params,
            index_folder=index_folder,
        )
    except Exception as exc:
        print(f'[reset_all] 索引构建失败 dataset={dataset_id} type={index_type}: {exc}')
        raise
    db.session.expire_all()
    built = db.session.get(AnnIndex, ann_index_id)
    print(f'  - {index_type} -> {built.status} ({built.index_file or "no file"})')

app = create_app()
with app.app_context():
    db.drop_all()
    db.create_all()

    admin = User(username='sysadmin', email='admin@scann.com', role='sysadmin')
    admin.set_password('pass123')
    db.session.add(admin)

    for role in ('visitor', 'normal', 'expert', 'labadmin'):
        u = User(username=role, email=f'{role}@scann.com', role=role)
        u.set_password('pass123')
        db.session.add(u)

    db.session.commit()

    upload_dir = app.config['UPLOAD_FOLDER']
    index_folder = app.config['INDEX_FOLDER']
    os.makedirs(upload_dir, exist_ok=True)
    if os.path.exists(index_folder):
        shutil.rmtree(index_folder)
    os.makedirs(index_folder, exist_ok=True)

    datasets = [
        ('外周血演示数据', 'demo_blood.h5ad'),
        ('人肺部实验数据', 'demo_lung.h5ad')
    ]

    for name, fname in datasets:
        path = os.path.join(upload_dir, fname)
        c, g, d = create_sample_h5ad(path, name)

        cell_types = ['TypeA', 'TypeB']
        obs_columns = ['cell_type']

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

        data = load_dataset(path)
        cache_dataset(ds.id, data)

        for index_type in ('hnsw', 'exact'):
            build_dataset_index(app, ds.id, index_type, d, index_folder)

    print("Database Reset Done with 2 Datasets and built indexes.")
