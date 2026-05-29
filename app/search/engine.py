import numpy as np
import time
import json
from ..models import db, Dataset, AnnIndex, QueryHistory
from ..data.loader import _dataset_cache, load_dataset, cache_dataset
from ..index.manager import search_index

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

    def search_by_cell_id(self, dataset_id: int, cell_id: str, index_id: int, top_k: int = 10, user_id: int = None):
        data = self._ensure_data(dataset_id)
        
        try:
            # 找到对应的行索引
            idx_in_data = data['cell_ids'].index(cell_id)
        except ValueError:
            raise ValueError(f"Cell ID '{cell_id}' 在数据集 {dataset_id} 中不存在")
            
        query_vector = data['vectors'][idx_in_data]
        
        return self._do_search(dataset_id, query_vector, index_id, top_k, user_id, 
                               query_type='cell_id', query_input=cell_id)

    def search_by_vector(self, dataset_id: int, vector, index_id: int, top_k: int = 10, user_id: int = None):
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)
        elif not isinstance(vector, np.ndarray):
            raise ValueError("Vector 必须是列表或 numpy 数组")
        
        # 简单记录向量的前几个元素作为输入
        query_input = f"Vector(dim={len(vector)})"
        
        return self._do_search(dataset_id, vector, index_id, top_k, user_id, 
                               query_type='vector', query_input=query_input)

    def search_random(self, dataset_id: int, index_id: int, top_k: int = 10, user_id: int = None):
        data = self._ensure_data(dataset_id)
        random_idx = np.random.randint(0, len(data['cell_ids']))
        cell_id = data['cell_ids'][random_idx]
        return self.search_by_cell_id(dataset_id, cell_id, index_id, top_k, user_id)

    def _do_search(self, dataset_id: int, query_vector: np.ndarray, index_id: int, 
                   top_k: int, user_id: int, query_type: str, query_input: str):
        ann_index = db.session.get(AnnIndex, index_id)
        if not ann_index or ann_index.dataset_id != dataset_id:
            raise ValueError(f"索引 {index_id} 不存在或与数据集不匹配")
        
        if ann_index.status != 'ready':
            raise ValueError(f"索引 {index_id} 尚未就绪 (状态: {ann_index.status})")

        t0 = time.time()
        # 调用 index/manager.py 中的搜索函数
        indices, distances = search_index(ann_index, query_vector, k=top_k)
        elapsed_ms = (time.time() - t0) * 1000

        data = _dataset_cache[dataset_id]
        obs  = data['obs']
        
        results = []
        for i, dist in zip(indices, distances):
            row = obs.iloc[i]
            res = {
                'cell_id': data['cell_ids'][i],
                'distance': float(dist),
                'cell_type': str(row.get('cell_type', 'unknown')),
                'metadata': {col: str(row[col]) for col in obs.columns[:10] if col in row} # 取前10个列防止太多
            }
            results.append(res)

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
            'results': results,
            'query_time_ms': elapsed_ms,
            'count': len(results)
        }
