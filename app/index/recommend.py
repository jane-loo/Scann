"""根据数据集规模推荐 ANN 构建/检索参数。"""
from __future__ import annotations

import math


def recommend_index_params(n_cells: int, index_type: str, n_dims: int = 30) -> dict:
    """
    返回推荐参数字典及说明。
    n_cells: 细胞数量
    index_type: hnsw | ivf_flat | ivf_pq | exact
    """
    n = max(1, int(n_cells or 1))
    dim = max(1, int(n_dims or 30))

    if index_type == 'exact':
        return {
            'params': {},
            'summary': 'Exact 索引无需 ANN 参数，作为精度基线。',
            'hints': ['适合小数据集或作为 Benchmark Ground Truth'],
        }

    if index_type == 'hnsw':
        if n < 500:
            M, ef_c, ef_s = 12, 100, 50
            tier = '小数据集'
        elif n < 5000:
            M, ef_c, ef_s = 16, 200, 100
            tier = '中等规模'
        else:
            M, ef_c, ef_s = 24, 400, 200
            tier = '大规模'
        return {
            'params': {
                'M': M,
                'ef_construction': ef_c,
                'ef_search': ef_s,
            },
            'summary': f'{tier}（{n} cells）：平衡精度与构建时间',
            'hints': [
                'M 越大图连接越密，Recall 更高但索引更大',
                'ef_construction 影响构建质量，ef_search 影响查询精度/延迟',
            ],
        }

    if index_type == 'ivf_flat':
        nlist = max(4, min(int(math.sqrt(n)), 256, max(1, n // 4)))
        nprobe = max(1, min(nlist, max(1, nlist // 10)))
        return {
            'params': {'nlist': nlist, 'nprobe': nprobe},
            'summary': f'IVF-Flat：nlist={nlist}，建议 nprobe={nprobe}',
            'hints': [
                'nlist 约为 sqrt(N)，过大在小数据集上训练不稳定',
                '增大 nprobe 提高 Recall，但查询变慢',
            ],
        }

    if index_type == 'ivf_pq':
        nlist = max(4, min(int(math.sqrt(n)), 128, max(1, n // 4)))
        nprobe = max(1, min(nlist, max(1, nlist // 8)))
        m_pq = 8 if dim % 8 == 0 else (4 if dim % 4 == 0 else 2)
        nbits = 8 if n >= 256 else 6 if n >= 64 else 4
        return {
            'params': {'nlist': nlist, 'nprobe': nprobe, 'm_pq': m_pq, 'nbits': nbits},
            'summary': f'IVF-PQ：省内存优先，nlist={nlist}, m_pq={m_pq}',
            'hints': [
                'PQ 压缩显著降低索引体积，Recall 通常低于 IVF-Flat',
                f'当前 dim={dim}，m_pq 需整除维度',
            ],
        }

    return {'params': {}, 'summary': '未知索引类型', 'hints': []}


def default_sweep_values(index_type: str) -> list[int]:
    """参数扫描默认值。"""
    if index_type == 'hnsw':
        return [32, 50, 100, 200, 400]
    return [1, 4, 8, 16, 32, 64]
