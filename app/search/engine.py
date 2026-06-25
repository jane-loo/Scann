import numpy as np
import time
import json
import pandas as pd
from ..models import db, Dataset, AnnIndex, QueryHistory
from ..data.loader import _dataset_cache, load_dataset, cache_dataset
from ..index.manager import search_index, index_is_usable, effective_index_status
from .filters import compute_expanded_k, row_matches_filters, build_result_metadata

class SearchEngine:
    def _ensure_data(self, dataset_id: int):
        """确保数据集向量在缓存中。"""
        if dataset_id not in _dataset_cache:
            dataset = db.session.get(Dataset, dataset_id)
            if not dataset:
                raise ValueError(f"数据集 {dataset_id} 不存在")
            data = load_dataset(dataset.file_path)
            cache_dataset(dataset_id, data)
        return _dataset_cache[dataset_id]

    def search_by_cell_id(self, dataset_id: int, cell_id: str, index_id: int, top_k: int = 10, user_id: int = None, filters: dict = None):
        data = self._ensure_data(dataset_id)
        
        try:
            # 找到对应的行索引
            idx_in_data = data['cell_ids'].index(cell_id)
        except ValueError:
            raise ValueError(f"Cell ID '{cell_id}' 在数据集 {dataset_id} 中不存在")
            
        query_vector = data['vectors'][idx_in_data]
        
        return self._do_search(dataset_id, query_vector, index_id, top_k, user_id, 
                               query_type='cell_id', query_input=cell_id, 
                               exclude_cell_id=cell_id, filters=filters)

    def search_by_vector(self, dataset_id: int, vector, index_id: int, top_k: int = 10, user_id: int = None, filters: dict = None):
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)
        elif not isinstance(vector, np.ndarray):
            raise ValueError("Vector 必须是列表或 numpy 数组")
        
        # 验证维度
        data = self._ensure_data(dataset_id)
        if len(vector) != data['vectors'].shape[1]:
            raise ValueError(f"输入向量维度({len(vector)})与数据集维度({data['vectors'].shape[1]})不匹配")

        query_input = f"Vector(dim={len(vector)})"
        
        return self._do_search(dataset_id, vector, index_id, top_k, user_id, 
                               query_type='vector', query_input=query_input, filters=filters)

    def search_random(self, dataset_id: int, index_id: int, top_k: int = 10, user_id: int = None, filters: dict = None):
        data = self._ensure_data(dataset_id)
        random_idx = np.random.randint(0, len(data['cell_ids']))
        cell_id = data['cell_ids'][random_idx]
        result = self.search_by_cell_id(
            dataset_id, cell_id, index_id, top_k, user_id, filters=filters
        )
        result['query_cell_id'] = cell_id
        result['query_type'] = 'random'
        return result

    def _do_search(self, dataset_id: int, query_vector: np.ndarray, index_id: int, 
                   top_k: int, user_id: int, query_type: str, query_input: str,
                   exclude_cell_id: str = None, filters: dict = None):
        ann_index = db.session.get(AnnIndex, index_id)
        if not ann_index or ann_index.dataset_id != dataset_id:
            raise ValueError(f"索引 {index_id} 不存在或与数据集不匹配")

        if not index_is_usable(ann_index):
            status = effective_index_status(ann_index)
            raise ValueError(f"索引 {index_id} 不可用 (状态: {status})，请先构建索引")

        # 为了排除自身或进行过滤，扩大搜索范围
        search_k = compute_expanded_k(top_k, exclude_self=bool(exclude_cell_id), filters=filters)

        t0 = time.time()
        # 调用 index/manager.py 中的搜索函数
        indices, distances = search_index(ann_index, query_vector, k=search_k)
        elapsed_ms = (time.time() - t0) * 1000

        data = _dataset_cache[dataset_id]
        obs  = data['obs']
        cell_ids = data['cell_ids']
        
        results = []
        for i, dist in zip(indices, distances):
            current_cell_id = cell_ids[i]
            
            # 1. 排除自身
            if exclude_cell_id and current_cell_id == exclude_cell_id:
                continue
                
            row = obs.iloc[i]
            
            if not row_matches_filters(row, filters):
                continue

            # 3. 计算相似度 score = 1 / (1 + distance)
            similarity = 1.0 / (1.0 + float(dist))
            
            res = {
                'rank': len(results) + 1,
                'cell_id': current_cell_id,
                'cell_index': current_cell_id,
                'distance': float(dist),
                'similarity': similarity,
                'cell_type': str(row.get('cell_type', '未知类型')),
                'metadata': build_result_metadata(row, obs),
            }
            results.append(res)
            
            # 达到 top_k 即停止
            if len(results) >= top_k:
                break

        # 记录查询历史
        history = QueryHistory(
            user_id     = user_id,
            dataset_id  = dataset_id,
            query_type  = query_type,
            query_input = query_input,
            index_type  = ann_index.index_type,
            top_k       = top_k,
            result_ids  = json.dumps([r['cell_id'] for r in results]),
            query_time  = elapsed_ms
        )
        db.session.add(history)
        db.session.commit()

        return {
            'query_time_ms': round(elapsed_ms, 2),
            'time_ms': round(elapsed_ms, 2),          # 兼容前端字段名
            'query_cell_id': query_input if query_type == 'cell_id' else 'Vector', # 兼容前端
            'top_k': len(results),
            'results': results
        }
