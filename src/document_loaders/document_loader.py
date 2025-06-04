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
    """文档加载和处理管理器"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        
        # 支持的文件类型映射
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
        """从指定目录加载所有支持的文档"""
        if directory_path is None:
            directory_path = settings.data_directory
            
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"目录不存在: {directory_path}")
            return []
        
        all_documents = []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.loader_mapping:
                try:
                    documents = self.load_single_file(str(file_path))
                    all_documents.extend(documents)
                    logger.info(f"成功加载文件: {file_path}, 文档片段数: {len(documents)}")
                except Exception as e:
                    logger.error(f"加载文件失败 {file_path}: {str(e)}")
        
        logger.info(f"总共加载 {len(all_documents)} 个文档片段")
        return all_documents
    
    def load_single_file(self, file_path: str) -> List[Document]:
        """加载单个文件"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_extension = path.suffix.lower()
        if file_extension not in self.loader_mapping:
            raise ValueError(f"不支持的文件类型: {file_extension}")
        
        loader_class = self.loader_mapping[file_extension]
        
        try:
            # 特殊处理需要编码参数的加载器
            if file_extension in ['.txt', '.md']:
                loader = loader_class(file_path, encoding='utf-8')
            else:
                loader = loader_class(file_path)
            
            documents = loader.load()
            
            # 为文档添加元数据
            for doc in documents:
                doc.metadata.update({
                    'source_file': str(path.name),
                    'file_path': str(path),
                    'file_type': file_extension
                })
            
            # 分割文档
            split_documents = self.text_splitter.split_documents(documents)
            
            return split_documents
            
        except Exception as e:
            logger.error(f"加载文件时出错 {file_path}: {str(e)}")
            raise
    
    def get_supported_extensions(self) -> List[str]:
        """获取支持的文件扩展名列表"""
        return list(self.loader_mapping.keys())
    
    def is_supported_file(self, file_path: str) -> bool:
        """检查文件是否为支持的类型"""
        return Path(file_path).suffix.lower() in self.loader_mapping 