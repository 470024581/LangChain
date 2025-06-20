from .vector_store import VectorStoreManager

# Dynamic vector store (optional import to avoid circular dependencies)
try:
    from .dynamic_vector_store import DynamicVectorStoreManager
    __all__ = ["VectorStoreManager", "DynamicVectorStoreManager"]
except ImportError:
    # If dependencies are not met, only export the basic version
    __all__ = ["VectorStoreManager"] 