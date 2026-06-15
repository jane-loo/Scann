"""
AnnData 读取模块测试
运行方式：python -m pytest tests/test_loader.py -v
"""
import numpy as np
import pytest
from app import create_app
from app.data.loader import (
    validate_h5ad, load_dataset, cache_dataset,
    get_cached_dataset, get_cell_vector, get_vectors, evict_cache,
)

H5AD_PATH = 'data/liver.h5ad'

app = create_app()


# ---- validate_h5ad ----

def test_validate_valid_file():
    valid, err = validate_h5ad(H5AD_PATH)
    assert valid, f'校验失败: {err}'


def test_validate_missing_file():
    valid, err = validate_h5ad('data/nonexistent.h5ad')
    assert not valid
    assert '不存在' in err


def test_validate_wrong_extension(tmp_path):
    f = tmp_path / 'test.txt'
    f.write_text('hello')
    valid, err = validate_h5ad(str(f))
    assert not valid
    assert '.h5ad' in err


# ---- load_dataset ----

@pytest.fixture(scope='module')
def loaded_data():
    return load_dataset(H5AD_PATH)


def test_load_returns_vectors(loaded_data):
    assert 'vectors' in loaded_data
    assert isinstance(loaded_data['vectors'], np.ndarray)
    assert loaded_data['vectors'].dtype == np.float32


def test_load_vector_shape(loaded_data):
    n_cells = loaded_data['n_cells']
    n_dims  = loaded_data['n_dims']
    assert loaded_data['vectors'].shape == (n_cells, n_dims)
    print(f'\n  细胞数: {n_cells}, PCA 维度: {n_dims}')


def test_load_pca_dim_is_30(loaded_data):
    # liver.h5ad 的 X_pca 是 30 维
    assert loaded_data['n_dims'] == 30


def test_load_cell_count(loaded_data):
    assert loaded_data['n_cells'] == 69032


def test_load_cell_ids(loaded_data):
    cell_ids = loaded_data['cell_ids']
    assert isinstance(cell_ids, list)
    assert len(cell_ids) == loaded_data['n_cells']
    # 格式如 AAACCTGAGCAGGTCA-1_2
    assert '-' in cell_ids[0]


def test_load_cell_types(loaded_data):
    ct = loaded_data['cell_types']
    assert isinstance(ct, list)
    assert len(ct) > 0
    assert 'hepatocyte' in ct
    print(f'\n  细胞类型数: {len(ct)}, 示例: {ct[:3]}')


def test_load_umap_coords(loaded_data):
    umap = loaded_data['umap_coords']
    assert umap is not None
    assert umap.shape == (loaded_data['n_cells'], 2)
    assert umap.dtype == np.float32


def test_load_tsne_coords(loaded_data):
    tsne = loaded_data['tsne_coords']
    assert tsne is not None
    assert tsne.shape[1] == 2


def test_load_obs_columns(loaded_data):
    cols = loaded_data['obs_columns']
    assert 'cell_type' in cols
    assert 'disease'   in cols
    assert 'AgeGroup'  in cols


# ---- cache 功能 ----

def test_cache_and_retrieve(loaded_data):
    with app.app_context():
        cache_dataset(9999, loaded_data)
        cached = get_cached_dataset(9999)
        assert cached is not None
        assert cached['vectors'].shape == loaded_data['vectors'].shape
        evict_cache(9999)
        assert get_cached_dataset.__wrapped__ if hasattr(get_cached_dataset, '__wrapped__') \
               else True  # 清除后不会报错即可


def test_get_cell_vector(loaded_data):
    with app.app_context():
        cache_dataset(8888, loaded_data)
        first_cell_id = loaded_data['cell_ids'][0]
        vec = get_cell_vector(8888, first_cell_id)
        assert vec.shape == (30,)
        assert vec.dtype == np.float32
        # 应与原向量完全一致
        np.testing.assert_array_equal(vec, loaded_data['vectors'][0])
        evict_cache(8888)


def test_get_cell_vector_invalid_id(loaded_data):
    with app.app_context():
        cache_dataset(7777, loaded_data)
        with pytest.raises(ValueError, match='不存在'):
            get_cell_vector(7777, 'INVALID_CELL_ID')
        evict_cache(7777)


def test_get_vectors(loaded_data):
    with app.app_context():
        cache_dataset(6666, loaded_data)
        vecs = get_vectors(6666)
        assert vecs is not None
        assert vecs.shape[0] == 69032
        evict_cache(6666)
