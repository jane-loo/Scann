import numpy as np

def recall_at_k(ann_results, gt_results, k=10):
    """
    计算 Recall@K。
    ann_results: ANN 搜索结果的 cell_id 列表 [id1, id2, ...]
    gt_results:  地面真值 (Exact Search) 的 cell_id 列表 [id1, id2, ...]
    k: 计算前 k 个结果的重合率
    """
    if not gt_results:
        return 0.0
    
    # 取前 k 个
    ann_k = set(ann_results[:k])
    gt_k  = set(gt_results[:k])
    
    intersection = ann_k.intersection(gt_k)
    return len(intersection) / len(gt_k)

def compute_qps(query_times_ms):
    """
    计算 QPS (Queries Per Second)。
    query_times_ms: 每次查询耗时的列表（单位：毫秒）
    """
    if not query_times_ms:
        return 0.0
    
    avg_time_ms = np.mean(query_times_ms)
    if avg_time_ms == 0:
        return 0.0
        
    return 1000.0 / avg_time_ms
