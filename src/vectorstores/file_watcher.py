import os
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, Set, Callable, Optional, List
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from ..config.settings import settings

logger = logging.getLogger(__name__)


class DocumentFileHandler(FileSystemEventHandler):
    """文档文件变化处理器"""
    
    # 支持的文档格式
    SUPPORTED_EXTENSIONS = {
        '.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.xls'
    }
    
    def __init__(self, on_file_added: Callable[[str], None] = None,
                 on_file_removed: Callable[[str], None] = None,
                 on_file_modified: Callable[[str], None] = None):
        """
        初始化文档文件处理器
        
        Args:
            on_file_added: 文件添加时的回调函数
            on_file_removed: 文件删除时的回调函数
            on_file_modified: 文件修改时的回调函数
        """
        super().__init__()
        self.on_file_added = on_file_added
        self.on_file_removed = on_file_removed
        self.on_file_modified = on_file_modified
        
        # 防止重复处理的缓存
        self._processing_files: Set[str] = set()
        
    def _is_supported_file(self, file_path: str) -> bool:
        """检查文件是否为支持的文档格式"""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def _should_process_file(self, file_path: str) -> bool:
        """检查是否应该处理该文件"""
        if not self._is_supported_file(file_path):
            return False
            
        # 避免处理临时文件
        filename = Path(file_path).name
        if filename.startswith('.') or filename.startswith('~'):
            return False
            
        # 避免重复处理
        if file_path in self._processing_files:
            return False
            
        return True
    
    def on_created(self, event: FileSystemEvent):
        """处理文件创建事件"""
        if event.is_directory:
            return
            
        file_path = event.src_path
        if self._should_process_file(file_path):
            logger.info(f"检测到新文件: {file_path}")
            self._processing_files.add(file_path)
            
            try:
                if self.on_file_added:
                    # 等待文件写入完成
                    time.sleep(1)
                    self.on_file_added(file_path)
            except Exception as e:
                logger.error(f"处理新文件失败 {file_path}: {str(e)}")
            finally:
                self._processing_files.discard(file_path)
    
    def on_deleted(self, event: FileSystemEvent):
        """处理文件删除事件"""
        if event.is_directory:
            return
            
        file_path = event.src_path
        if self._is_supported_file(file_path):
            logger.info(f"检测到文件删除: {file_path}")
            
            try:
                if self.on_file_removed:
                    self.on_file_removed(file_path)
            except Exception as e:
                logger.error(f"处理文件删除失败 {file_path}: {str(e)}")
    
    def on_modified(self, event: FileSystemEvent):
        """处理文件修改事件"""
        if event.is_directory:
            return
            
        file_path = event.src_path
        if self._should_process_file(file_path):
            logger.info(f"检测到文件修改: {file_path}")
            self._processing_files.add(file_path)
            
            try:
                if self.on_file_modified:
                    # 等待文件写入完成
                    time.sleep(1)
                    self.on_file_modified(file_path)
            except Exception as e:
                logger.error(f"处理文件修改失败 {file_path}: {str(e)}")
            finally:
                self._processing_files.discard(file_path)


class FileSystemWatcher:
    """文件系统监控器"""
    
    def __init__(self, watch_directory: str = None):
        """
        初始化文件系统监控器
        
        Args:
            watch_directory: 要监控的目录路径
        """
        self.watch_directory = watch_directory or settings.vector_watch_directory
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[DocumentFileHandler] = None
        self.is_running = False
        
        # 回调函数
        self._file_added_callbacks: List[Callable[[str], None]] = []
        self._file_removed_callbacks: List[Callable[[str], None]] = []
        self._file_modified_callbacks: List[Callable[[str], None]] = []
        
        logger.info(f"文件系统监控器初始化，监控目录: {self.watch_directory}")
    
    def add_file_added_callback(self, callback: Callable[[str], None]):
        """添加文件添加事件回调"""
        self._file_added_callbacks.append(callback)
    
    def add_file_removed_callback(self, callback: Callable[[str], None]):
        """添加文件删除事件回调"""
        self._file_removed_callbacks.append(callback)
    
    def add_file_modified_callback(self, callback: Callable[[str], None]):
        """添加文件修改事件回调"""
        self._file_modified_callbacks.append(callback)
    
    def _on_file_added(self, file_path: str):
        """处理文件添加事件"""
        logger.info(f"文件添加事件: {file_path}")
        for callback in self._file_added_callbacks:
            try:
                callback(file_path)
            except Exception as e:
                logger.error(f"文件添加回调执行失败: {str(e)}")
    
    def _on_file_removed(self, file_path: str):
        """处理文件删除事件"""
        logger.info(f"文件删除事件: {file_path}")
        for callback in self._file_removed_callbacks:
            try:
                callback(file_path)
            except Exception as e:
                logger.error(f"文件删除回调执行失败: {str(e)}")
    
    def _on_file_modified(self, file_path: str):
        """处理文件修改事件"""
        logger.info(f"文件修改事件: {file_path}")
        for callback in self._file_modified_callbacks:
            try:
                callback(file_path)
            except Exception as e:
                logger.error(f"文件修改回调执行失败: {str(e)}")
    
    def start(self):
        """启动文件系统监控"""
        if self.is_running:
            logger.warning("文件系统监控器已在运行")
            return
        
        # 确保监控目录存在
        watch_path = Path(self.watch_directory)
        if not watch_path.exists():
            logger.error(f"监控目录不存在: {self.watch_directory}")
            return
        
        # 创建事件处理器
        self.event_handler = DocumentFileHandler(
            on_file_added=self._on_file_added,
            on_file_removed=self._on_file_removed,
            on_file_modified=self._on_file_modified
        )
        
        # 创建观察者
        self.observer = Observer()
        self.observer.schedule(
            self.event_handler,
            self.watch_directory,
            recursive=True
        )
        
        # 启动监控
        self.observer.start()
        self.is_running = True
        
        logger.info(f"文件系统监控器已启动，监控目录: {self.watch_directory}")
    
    def stop(self):
        """停止文件系统监控"""
        if not self.is_running:
            logger.warning("文件系统监控器未在运行")
            return
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        self.event_handler = None
        self.is_running = False
        
        logger.info("文件系统监控器已停止")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop() 