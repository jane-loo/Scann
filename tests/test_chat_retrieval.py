"""元数据过滤单元测试。"""
import pandas as pd

from app.search.filters import compute_expanded_k, row_matches_filters, build_result_metadata


class TestMetadataFilters:
    def test_compute_expanded_k_without_filters(self):
        assert compute_expanded_k(10) == 10

    def test_compute_expanded_k_with_filters(self):
        assert compute_expanded_k(10, filters={'cell_type': ['T cell']}) == 100

    def test_row_matches_filters(self):
        row = pd.Series({'cell_type': 'T cell', 'disease': 'normal'})
        assert row_matches_filters(row, {'cell_type': ['T cell']})
        assert not row_matches_filters(row, {'cell_type': ['B cell']})

    def test_build_result_metadata(self):
        obs = pd.DataFrame({'cell_type': ['T cell'], 'disease': ['normal']})
        meta = build_result_metadata(obs.iloc[0], obs)
        assert meta['cell_type'] == 'T cell'
        assert meta['disease'] == 'normal'
