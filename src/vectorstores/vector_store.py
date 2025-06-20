import os
from typing import List, Optional
from pathlib import Path
import logging
import pickle

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from ..config.settings import settings
from ..document_loaders.document_loader import DocumentLoaderManager

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Vector store manager"""
    
    def __init__(self, use_openai_embeddings: bool = False):
        """
        Initialize vector store manager
        
        Args:
            use_openai_embeddings: Whether to use OpenAI embeddings, otherwise use HuggingFace
        """
        self.vector_store_path = Path(settings.vector_store_path)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        if use_openai_embeddings:
            try:
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=settings.openrouter_api_key,
                    openai_api_base=settings.openrouter_api_base
                )
                logger.info("Using OpenAI embeddings")
            except Exception as e:
                logger.warning(f"OpenAI embeddings initialization failed: {e}, switching to HuggingFace")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=settings.embedding_model_name
                )
                logger.info("Using HuggingFace embeddings")
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name
            )
            logger.info("Using HuggingFace embeddings")
        
        self.vector_store: Optional[FAISS] = None
        self.document_loader = DocumentLoaderManager()
    
    def create_vector_store(self, documents: List[Document] = None) -> FAISS:
        """Create vector store"""
        if documents is None:
            logger.info("Loading documents...")
            documents = self.document_loader.load_documents_from_directory()
        
        if not documents:
            raise ValueError("No available documents found")
        
        logger.info(f"Creating vector store, document count: {len(documents)}")
        
        try:
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            logger.info("Vector store created successfully")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise
    
    def save_vector_store(self, store_name: str = "default") -> None:
        """Save vector store to local"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        store_path = self.vector_store_path / store_name
        store_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.vector_store.save_local(str(store_path))
            logger.info(f"Vector store saved to: {store_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise
    
    def load_vector_store(self, store_name: str = "default") -> FAISS:
        """Load vector store from local"""
        store_path = self.vector_store_path / store_name
        
        if not store_path.exists():
            logger.warning(f"Vector store does not exist: {store_path}")
            return None
        
        try:
            self.vector_store = FAISS.load_local(
                str(store_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded from {store_path}")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise
    
    def get_or_create_vector_store(self, store_name: str = "default", force_recreate: bool = False) -> FAISS:
        """Get or create vector store"""
        if force_recreate or not self._vector_store_exists(store_name):
            logger.info("Creating new vector store...")
            self.create_vector_store()
            self.save_vector_store(store_name)
        else:
            logger.info("Loading existing vector store...")
            self.load_vector_store(store_name)
        
        return self.vector_store
    
    def _vector_store_exists(self, store_name: str = "default") -> bool:
        """Check if vector store exists"""
        store_path = self.vector_store_path / store_name
        return store_path.exists() and (store_path / "index.faiss").exists()
    
    def get_retriever(self, k: int = 4, search_type: str = "similarity"):
        """Get retriever"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized, please call get_or_create_vector_store() first")
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Similarity search"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        self.vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def delete_vector_store(self, store_name: str = "default") -> None:
        """Delete vector store"""
        store_path = self.vector_store_path / store_name
        if store_path.exists():
            import shutil
            shutil.rmtree(store_path)
            logger.info(f"Deleted vector store: {store_path}")
        
        if self.vector_store is not None:
            self.vector_store = None
    
    def delete_documents_by_ids(self, ids: List[str]) -> bool:
        """Delete documents by IDs"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        try:
            result = self.vector_store.delete(ids)
            logger.info(f"Deleted {len(ids)} documents")
            return result
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise
    
    def delete_documents_by_source(self, source_path: str) -> bool:
        """Delete related documents by source file path"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        # Find all document IDs for this source file
        doc_ids = []
        for doc_id, doc in self.vector_store.docstore._dict.items():
            if hasattr(doc, 'metadata') and doc.metadata.get('source') == source_path:
                doc_ids.append(doc_id)
        
        if not doc_ids:
            logger.warning(f"No related documents found for source file {source_path}")
            return False
        
        try:
            result = self.vector_store.delete(doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents for source file {source_path}")
            return result
        except Exception as e:
            logger.error(f"Failed to delete source file documents: {str(e)}")
            raise
    
    def get_document_ids_by_source(self, source_path: str) -> List[str]:
        """Get all document IDs for specified source file"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        doc_ids = []
        for doc_id, doc in self.vector_store.docstore._dict.items():
            if hasattr(doc, 'metadata') and doc.metadata.get('source') == source_path:
                doc_ids.append(doc_id)
        
        return doc_ids
    
    def rebuild_from_directory(self, force_rebuild: bool = True) -> FAISS:
        """Rescan directory and rebuild vector store"""
        logger.info("Rescanning document directory...")
        documents = self.document_loader.load_documents_from_directory()
        
        if force_rebuild:
            logger.info("Force rebuilding vector store...")
            self.create_vector_store(documents)
        else:
            # Incremental update logic can be implemented here
            logger.info("Incremental updating vector store...")
            if self.vector_store is None:
                self.create_vector_store(documents)
            else:
                # Logic for incremental updates needs to be implemented here
                # Compare existing documents with new documents, add new ones, remove non-existent ones
                pass
        
        return self.vector_store 