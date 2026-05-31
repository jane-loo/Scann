"""
app.index 公共接口（第五步：为成员 B 暴露内部接口）。

成员 B 在检索引擎中只需：
    from app.index import get_index_object, get_cell_vector, get_obs_dataframe

即可获取索引对象和数据访问工具，无需了解内部实现细节。
"""

# ── 索引对象获取 ──────────────────────────────────────────────────────────────
from .manager import get_index_object          # noqa: F401

# ── 构建/管理接口（内部及路由使用）────────────────────────────────────────────
from .manager import (                         # noqa: F401
    build_index_async,
    build_index_sync,
    build_joint_index_async,
    build_joint_index_sync,
    search_index,
    search_joint_index,
    evict_index_cache,
)

# ── 数据访问接口转发（供 Member B 无需跨模块导入）───────────────────────────
from ..data.loader import (                    # noqa: F401
    get_vectors,
    get_cell_vector,
    get_obs_dataframe,
    get_cached_dataset,
)

# ── 底层索引类（供评测/调试使用）────────────────────────────────────────────
from .hnsw_index  import HNSWIndex             # noqa: F401
from .faiss_index import FaissIndex            # noqa: F401

__all__ = [
    # 高层入口（Member B 主要用这些）
    'get_index_object',
    'get_cell_vector',
    'get_obs_dataframe',
    'get_cached_dataset',
    'get_vectors',
    # 构建/管理
    'build_index_async',
    'build_index_sync',
    'build_joint_index_async',
    'build_joint_index_sync',
    'search_index',
    'search_joint_index',
    'evict_index_cache',
    # 底层类
    'HNSWIndex',
    'FaissIndex',
]
