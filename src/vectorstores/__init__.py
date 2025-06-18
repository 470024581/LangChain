from .vector_store import VectorStoreManager

# 动态向量存储（可选导入，避免循环依赖）
try:
    from .dynamic_vector_store import DynamicVectorStoreManager
    __all__ = ["VectorStoreManager", "DynamicVectorStoreManager"]
except ImportError:
    # 如果依赖不满足，只导出基础版本
    __all__ = ["VectorStoreManager"] 