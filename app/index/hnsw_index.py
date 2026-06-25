"""
HNSWLIB 向量索引封装。

支持度量：
  - 'l2'     → 欧氏距离（hnswlib space='l2'）
  - 'cosine' → 余弦距离（hnswlib space='cosine'）
  - 'ip'     → 内积（hnswlib space='ip'）
"""
import numpy as np
import hnswlib


class HNSWIndex:
    """轻量封装 hnswlib.Index，统一 build / save / load / search 接口。"""

    _SPACE_MAP = {'l2': 'l2', 'cosine': 'cosine', 'ip': 'ip'}

    def __init__(self):
        self.index: hnswlib.Index | None = None
        self.dim:   int | None           = None
        self.space: str                  = 'l2'

    # ──────────────────────────────────────────────
    # 构建
    # ──────────────────────────────────────────────

    def build(self,
              vectors:         np.ndarray,
              metric:          str = 'l2',
              M:               int = 16,
              ef_construction: int = 200) -> None:
        """
        从向量矩阵构建索引。

        Args:
            vectors:         shape (n, dim), dtype float32
            metric:          'l2' | 'cosine' | 'ip'
            M:               每层最大邻接边数，越大精度越高但内存越多
            ef_construction: 构建时的动态搜索宽度，越大质量越好但越慢
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f'vectors 必须是二维数组，当前形状: {vectors.shape}')
        n, dim = vectors.shape

        space = self._SPACE_MAP.get(metric, 'l2')
        self.dim   = dim
        self.space = space

        index = hnswlib.Index(space=space, dim=dim)
        index.init_index(max_elements=n, M=M, ef_construction=ef_construction)
        index.add_items(vectors, list(range(n)))
        # 搜索时的动态宽度，一般取 ef_construction / 2，但不低于 50
        index.set_ef(max(50, ef_construction // 2))
        self.index = index

    # ──────────────────────────────────────────────
    # 持久化
    # ──────────────────────────────────────────────

    def save(self, path: str) -> None:
        """将索引保存到文件（hnswlib 原生二进制格式）。

        hnswlib 的 C++ 实现在 Windows 上无法处理路径中的非 ASCII 字符（如中文目录名）。
        解决方案：先保存到系统临时目录（纯 ASCII 路径），再用 Python 的 shutil.move 移到目标。
        """
        if self.index is None:
            raise RuntimeError('索引尚未构建，请先调用 build()')

        # 路径全为 ASCII 时直接保存，无需绕路
        if all(ord(c) < 128 for c in path):
            self.index.save_index(path)
            return

        import tempfile, shutil, os
        fd, tmp_path = tempfile.mkstemp(suffix='.bin')
        os.close(fd)
        try:
            self.index.save_index(tmp_path)
            shutil.move(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def load(self, path: str, dim: int, space: str = 'l2') -> None:
        """
        从文件加载索引。hnswlib 同样不支持非 ASCII 路径，先复制到临时文件再加载。

        Args:
            path:  索引文件路径
            dim:   向量维度（必须与构建时一致）
            space: 度量空间，'l2' | 'cosine' | 'ip'
        """
        self.dim   = dim
        self.space = space
        index = hnswlib.Index(space=space, dim=dim)

        if all(ord(c) < 128 for c in path):
            index.load_index(path, max_elements=0)
        else:
            import tempfile, shutil, os
            fd, tmp_path = tempfile.mkstemp(suffix='.bin')
            os.close(fd)
            try:
                shutil.copy2(path, tmp_path)
                index.load_index(tmp_path, max_elements=0)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        self.index = index

    # ──────────────────────────────────────────────
    # 搜索
    # ──────────────────────────────────────────────

    def search(self,
               query: np.ndarray,
               k:     int = 10,
               ef:    int | None = None) -> tuple[list[int], list[float]]:
        """
        kNN 搜索。

        Args:
            query: shape (dim,) 或 (1, dim)，dtype float32
            k:     返回的最近邻数量

        Returns:
            (indices, distances)：
              - indices   为向量整数位置列表（与构建时 add_items 的 label 对应）
              - distances 为对应距离列表
        """
        if self.index is None:
            raise RuntimeError('索引未就绪，请先 build() 或 load()')

        if ef is not None:
            self.index.set_ef(max(1, int(ef)))

        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        k = min(k, self.index.element_count)
        labels, distances = self.index.knn_query(query, k=k)
        return labels[0].tolist(), distances[0].tolist()

    # ──────────────────────────────────────────────
    # 属性
    # ──────────────────────────────────────────────

    @property
    def element_count(self) -> int:
        if self.index is None:
            return 0
        return self.index.element_count
