import os
from typing import List, Optional
from pathlib import Path
import logging
import pickle

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from ..config.settings import settings
from ..document_loaders.document_loader import DocumentLoaderManager

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """向量存储管理器"""
    
    def __init__(self, use_openai_embeddings: bool = True):
        """
        初始化向量存储管理器
        
        Args:
            use_openai_embeddings: 是否使用OpenAI embeddings，否则使用HuggingFace
        """
        self.vector_store_path = Path(settings.vector_store_path)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化嵌入模型
        if use_openai_embeddings:
            try:
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=settings.openrouter_api_key,
                    openai_api_base=settings.openrouter_api_base
                )
                logger.info("使用OpenAI embeddings")
            except Exception as e:
                logger.warning(f"OpenAI embeddings初始化失败: {e}, 切换到HuggingFace")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("使用HuggingFace embeddings")
        
        self.vector_store: Optional[FAISS] = None
        self.document_loader = DocumentLoaderManager()
    
    def create_vector_store(self, documents: List[Document] = None) -> FAISS:
        """创建向量存储"""
        if documents is None:
            logger.info("加载文档...")
            documents = self.document_loader.load_documents_from_directory()
        
        if not documents:
            raise ValueError("没有找到可用的文档")
        
        logger.info(f"创建向量存储，文档数量: {len(documents)}")
        
        try:
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            logger.info("向量存储创建成功")
            return self.vector_store
        except Exception as e:
            logger.error(f"创建向量存储失败: {str(e)}")
            raise
    
    def save_vector_store(self, store_name: str = "default") -> None:
        """保存向量存储到本地"""
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        
        store_path = self.vector_store_path / store_name
        store_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.vector_store.save_local(str(store_path))
            logger.info(f"向量存储已保存到: {store_path}")
        except Exception as e:
            logger.error(f"保存向量存储失败: {str(e)}")
            raise
    
    def load_vector_store(self, store_name: str = "default") -> FAISS:
        """从本地加载向量存储"""
        store_path = self.vector_store_path / store_name
        
        if not store_path.exists():
            logger.warning(f"向量存储不存在: {store_path}")
            return None
        
        try:
            self.vector_store = FAISS.load_local(
                str(store_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"向量存储已从 {store_path} 加载")
            return self.vector_store
        except Exception as e:
            logger.error(f"加载向量存储失败: {str(e)}")
            raise
    
    def get_or_create_vector_store(self, store_name: str = "default", force_recreate: bool = False) -> FAISS:
        """获取或创建向量存储"""
        if force_recreate or not self._vector_store_exists(store_name):
            logger.info("创建新的向量存储...")
            self.create_vector_store()
            self.save_vector_store(store_name)
        else:
            logger.info("加载现有向量存储...")
            self.load_vector_store(store_name)
        
        return self.vector_store
    
    def _vector_store_exists(self, store_name: str = "default") -> bool:
        """检查向量存储是否存在"""
        store_path = self.vector_store_path / store_name
        return store_path.exists() and (store_path / "index.faiss").exists()
    
    def get_retriever(self, k: int = 4, search_type: str = "similarity"):
        """获取检索器"""
        if self.vector_store is None:
            raise ValueError("向量存储未初始化，请先调用 get_or_create_vector_store()")
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """相似性搜索"""
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def add_documents(self, documents: List[Document]) -> None:
        """向现有向量存储添加文档"""
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        
        self.vector_store.add_documents(documents)
        logger.info(f"已添加 {len(documents)} 个文档到向量存储")
    
    def delete_vector_store(self, store_name: str = "default") -> None:
        """删除向量存储"""
        store_path = self.vector_store_path / store_name
        if store_path.exists():
            import shutil
            shutil.rmtree(store_path)
            logger.info(f"已删除向量存储: {store_path}")
        
        if self.vector_store is not None:
            self.vector_store = None 