"""
FAISS 向量索引封装。

支持索引类型：
  - 'exact'    → IndexFlatL2 / IndexFlatIP（精确搜索，无需训练）
  - 'ivf_flat' → IndexIVFFlat（倒排 + 平坦量化，需训练）
  - 'ivf_pq'   → IndexIVFPQ（倒排 + 乘积量化，需训练，内存最小）

支持度量：
  - 'l2'     → 欧氏距离
  - 'cosine' → 余弦相似度（归一化后用内积近似）
"""
import numpy as np
import faiss


class FaissIndex:
    """统一封装 FAISS 索引，提供 build / save / load / search 接口。"""

    def __init__(self):
        self.index:      faiss.Index | None = None
        self.dim:        int | None         = None
        self.index_type: str                = 'exact'
        self.metric:     str                = 'l2'

    # ──────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2 归一化（原地），cosine 度量时使用。"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vectors / norms).astype(np.float32)

    def _flat_quantizer(self, dim: int):
        if self.metric == 'cosine':
            return faiss.IndexFlatIP(dim)
        return faiss.IndexFlatL2(dim)

    @staticmethod
    def _adapt_ivf_nlist(n: int, nlist: int) -> int:
        """IVF 聚类数不能超过训练向量数，小数据集需收紧 nlist。"""
        return max(1, min(nlist, max(1, n // 4), n))

    @staticmethod
    def _adapt_pq_m(dim: int, m_pq: int) -> int:
        """PQ 子空间数必须整除向量维度。"""
        m_pq = min(m_pq, dim)
        while dim % m_pq != 0 and m_pq > 1:
            m_pq -= 1
        if dim % m_pq != 0:
            raise ValueError(f'ivf_pq: 无法为 dim={dim} 选择合法的 m_pq')
        return m_pq

    @staticmethod
    def _adapt_pq_nbits(n: int, nbits: int) -> int:
        """
        PQ 训练对每个子空间做 k-means，k = 2^nbits。
        训练向量数 n 必须 >= k，否则 FAISS 报 nx >= k 错误。
        """
        nbits = min(max(nbits, 4), 8)
        while nbits > 4 and (1 << nbits) > n:
            nbits -= 1
        min_required = 1 << nbits
        if n < min_required:
            raise ValueError(
                f'ivf_pq 至少需要 {min_required} 个向量（PQ 码本 2^{nbits}），'
                f'当前 n={n}。请增大数据集或改用 ivf_flat / exact。'
            )
        return nbits

    # ──────────────────────────────────────────────
    # 构建
    # ──────────────────────────────────────────────

    def build(self,
              vectors:    np.ndarray,
              index_type: str = 'exact',
              metric:     str = 'l2',
              nlist:      int = 100,
              m_pq:       int = 8,
              nbits:      int = 8) -> None:
        """
        构建 FAISS 索引。

        Args:
            vectors:    shape (n, dim), dtype float32
            index_type: 'exact' | 'ivf_flat' | 'ivf_pq'
            metric:     'l2' | 'cosine'
            nlist:      IVF 聚类中心数（ivf_flat / ivf_pq）
            m_pq:       PQ 子空间数（ivf_pq；必须整除 dim）
            nbits:      每子空间比特数（ivf_pq，通常 8）
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f'vectors 必须是二维数组，当前形状: {vectors.shape}')
        n, dim = vectors.shape

        self.dim        = dim
        self.index_type = index_type
        self.metric     = metric

        # cosine：归一化后用内积
        if metric == 'cosine':
            vectors = self._normalize(vectors)

        faiss_metric = (faiss.METRIC_INNER_PRODUCT
                        if metric == 'cosine'
                        else faiss.METRIC_L2)

        if index_type == 'exact':
            idx = (faiss.IndexFlatIP(dim)
                   if metric == 'cosine'
                   else faiss.IndexFlatL2(dim))

        elif index_type == 'ivf_flat':
            nlist = self._adapt_ivf_nlist(n, nlist)
            quantizer = self._flat_quantizer(dim)
            idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss_metric)
            idx.train(vectors)

        elif index_type == 'ivf_pq':
            nlist = self._adapt_ivf_nlist(n, nlist)
            m_pq  = self._adapt_pq_m(dim, m_pq)
            nbits = self._adapt_pq_nbits(n, nbits)
            quantizer = self._flat_quantizer(dim)
            idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, nbits, faiss_metric)
            idx.train(vectors)

        else:
            raise ValueError(f'不支持的 index_type: {index_type}，'
                             f'可选: exact, ivf_flat, ivf_pq')

        idx.add(vectors)
        self.index = idx

    # ──────────────────────────────────────────────
    # 持久化
    # ──────────────────────────────────────────────

    def save(self, path: str) -> None:
        if self.index is None:
            raise RuntimeError('索引尚未构建，请先调用 build()')
        # 修复 Windows 下非 ASCII 路径导致的 faiss 写入失败（使用 Python 文件流包装）
        try:
            # serialize_index 返回的是 numpy array (uint8)
            chunk = faiss.serialize_index(self.index)
            with open(path, 'wb') as f:
                f.write(chunk.tobytes())
        except Exception as e:
            # 如果 serialize 不支持，降级使用原生方法（可能在非 ASCII 路径下失败）
            faiss.write_index(self.index, path)

    def load(self, path: str, metric: str = 'l2') -> None:
        """
        从文件加载索引。
        """
        try:
            with open(path, 'rb') as f:
                data = f.read()
            # deserialize_index 需要 uint8 的 numpy array
            arr = np.frombuffer(data, dtype=np.uint8)
            self.index = faiss.deserialize_index(arr)
        except Exception as e:
            # 降级
            self.index = faiss.read_index(path)
        self.dim    = self.index.d
        self.metric = metric

    # ──────────────────────────────────────────────
    # 搜索
    # ──────────────────────────────────────────────

    def search(self,
               query:  np.ndarray,
               k:      int = 10,
               nprobe: int = 10) -> tuple[list[int], list[float]]:
        """
        kNN 搜索。

        Args:
            query:  shape (dim,) 或 (1, dim)，dtype float32
            k:      返回的最近邻数量
            nprobe: IVF 搜索时探测的聚类数（仅 ivf_flat / ivf_pq 生效）

        Returns:
            (indices, distances)：
              - indices   为向量整数位置列表（−1 表示无效，应过滤掉）
              - distances 为对应距离/相似度列表
        """
        if self.index is None:
            raise RuntimeError('索引未就绪，请先 build() 或 load()')

        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self.metric == 'cosine':
            query = self._normalize(query)

        # IVF 索引支持 nprobe 调整
        if hasattr(self.index, 'nprobe'):
            nlist = getattr(self.index, 'nlist', nprobe)
            self.index.nprobe = max(1, min(nprobe, nlist))

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query, k)

        idx_list  = indices[0].tolist()
        dist_list = distances[0].tolist()

        # 过滤 FAISS 填充的 -1（向量不足时）
        valid = [(i, d) for i, d in zip(idx_list, dist_list) if i >= 0]
        if valid:
            idx_list, dist_list = zip(*valid)
            return list(idx_list), list(dist_list)
        return [], []

    # ──────────────────────────────────────────────
    # 属性
    # ──────────────────────────────────────────────

    @property
    def element_count(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal
