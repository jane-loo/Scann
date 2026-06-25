"""索引参数推荐与扫描默认值（直接加载模块，避免 Flask 依赖）。"""
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'recommend', _ROOT / 'app' / 'index' / 'recommend.py',
)
recommend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recommend)


def test_hnsw_recommend_scales_with_n_cells():
    small = recommend.recommend_index_params(200, 'hnsw')
    large = recommend.recommend_index_params(10000, 'hnsw')
    assert small['params']['M'] <= large['params']['M']
    assert small['params']['ef_search'] <= large['params']['ef_search']


def test_ivf_flat_nlist_reasonable():
    rec = recommend.recommend_index_params(5000, 'ivf_flat', 30)
    nlist = rec['params']['nlist']
    assert 4 <= nlist <= 256
    assert rec['params']['nprobe'] <= nlist


def test_runtime_playground_config_hnsw():
    cfg = recommend.runtime_playground_config('hnsw', {'ef_search': 80})
    assert cfg['param_name'] == 'ef_search'
    assert cfg['default'] == 80
    assert 'fast' in cfg['presets']


def test_runtime_playground_config_ivf():
    cfg = recommend.runtime_playground_config('ivf_flat', {'nlist': 64})
    assert cfg['param_name'] == 'nprobe'
    assert cfg['max'] <= 64
