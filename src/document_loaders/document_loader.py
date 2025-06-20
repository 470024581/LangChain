import os
from typing import List, Dict, Any
from pathlib import Path
import logging

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config.settings import settings

logger = logging.getLogger(__name__)


class DocumentLoaderManager:
    """Document loading and processing manager"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        
        # Supported file type mapping
        self.loader_mapping = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.doc': Docx2txtLoader,
            '.txt': TextLoader,
            '.md': TextLoader,
            '.csv': CSVLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader
        }
    
    def load_documents_from_directory(self, directory_path: str = None) -> List[Document]:
        """Load all supported documents from specified directory"""
        if directory_path is None:
            directory_path = settings.data_directory
            
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        all_documents = []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.loader_mapping:
                try:
                    documents = self.load_single_file(str(file_path))
                    all_documents.extend(documents)
                    logger.info(f"Successfully loaded file: {file_path}, document chunks: {len(documents)}")
                except Exception as e:
                    logger.error(f"Failed to load file {file_path}: {str(e)}")
        
        logger.info(f"Total loaded {len(all_documents)} document chunks")
        return all_documents
    
    def load_single_file(self, file_path: str) -> List[Document]:
        """Load single file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        file_extension = path.suffix.lower()
        if file_extension not in self.loader_mapping:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        loader_class = self.loader_mapping[file_extension]
        
        try:
            # Special handling for loaders that need encoding parameter
            if file_extension in ['.txt', '.md']:
                loader = loader_class(file_path, encoding='utf-8')
            else:
                loader = loader_class(file_path)
            
            documents = loader.load()
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    'source_file': str(path.name),
                    'file_path': str(path),
                    'file_type': file_extension
                })
            
            # Split documents
            split_documents = self.text_splitter.split_documents(documents)
            
            return split_documents
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return list(self.loader_mapping.keys())
    
    def is_supported_file(self, file_path: str) -> bool:
        """Check if file is a supported type"""
        return Path(file_path).suffix.lower() in self.loader_mapping 