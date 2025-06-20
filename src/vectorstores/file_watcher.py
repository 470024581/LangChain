from __future__ import annotations

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
    """Document file change handler"""
    
    # Supported document formats
    SUPPORTED_EXTENSIONS = {
        '.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.xls'
    }
    
    def __init__(self, on_file_added: Optional[Callable[[str], None]] = None,
                 on_file_removed: Optional[Callable[[str], None]] = None,
                 on_file_modified: Optional[Callable[[str], None]] = None):
        """
        Initialize document file handler
        
        Args:
            on_file_added: Callback function when file is added
            on_file_removed: Callback function when file is removed
            on_file_modified: Callback function when file is modified
        """
        super().__init__()
        self.on_file_added = on_file_added
        self.on_file_removed = on_file_removed
        self.on_file_modified = on_file_modified
        
        # Cache to prevent duplicate processing
        self._processing_files: Set[str] = set()
        
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file is a supported document format"""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def _should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed"""
        if not self._is_supported_file(file_path):
            return False
            
        # Avoid processing temporary files
        filename = Path(file_path).name
        if filename.startswith('.') or filename.startswith('~'):
            return False
            
        # Avoid duplicate processing
        if file_path in self._processing_files:
            return False
            
        return True
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation event"""
        if event.is_directory:
            return
            
        file_path = event.src_path
        if self._should_process_file(file_path):
            logger.info(f"Detected new file: {file_path}")
            self._processing_files.add(file_path)
            
            try:
                if self.on_file_added:
                    # Wait for file write completion
                    time.sleep(1)
                    self.on_file_added(file_path)
            except Exception as e:
                logger.error(f"Failed to process new file {file_path}: {str(e)}")
            finally:
                self._processing_files.discard(file_path)
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion event"""
        if event.is_directory:
            return
            
        file_path = event.src_path
        if self._is_supported_file(file_path):
            logger.info(f"Detected file deletion: {file_path}")
            
            try:
                if self.on_file_removed:
                    self.on_file_removed(file_path)
            except Exception as e:
                logger.error(f"Failed to process file deletion {file_path}: {str(e)}")
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification event"""
        if event.is_directory:
            return
            
        file_path = event.src_path
        if self._should_process_file(file_path):
            logger.info(f"Detected file modification: {file_path}")
            self._processing_files.add(file_path)
            
            try:
                if self.on_file_modified:
                    # Wait for file write completion
                    time.sleep(1)
                    self.on_file_modified(file_path)
            except Exception as e:
                logger.error(f"Failed to process file modification {file_path}: {str(e)}")
            finally:
                self._processing_files.discard(file_path)


class FileSystemWatcher:
    """File system watcher"""
    
    def __init__(self, watch_directory: Optional[str] = None):
        """
        Initialize file system watcher
        
        Args:
            watch_directory: Directory path to watch
        """
        self.watch_directory = watch_directory or settings.vector_watch_directory
        self.observer = None
        self.event_handler = None
        self.is_running = False
        
        # Callback functions
        self._file_added_callbacks: List[Callable[[str], None]] = []
        self._file_removed_callbacks: List[Callable[[str], None]] = []
        self._file_modified_callbacks: List[Callable[[str], None]] = []
        
        logger.info(f"File system watcher initialized, watching directory: {self.watch_directory}")
    
    def add_file_added_callback(self, callback: Callable[[str], None]):
        """Add file addition event callback"""
        self._file_added_callbacks.append(callback)
    
    def add_file_removed_callback(self, callback: Callable[[str], None]):
        """Add file deletion event callback"""
        self._file_removed_callbacks.append(callback)
    
    def add_file_modified_callback(self, callback: Callable[[str], None]):
        """Add file modification event callback"""
        self._file_modified_callbacks.append(callback)
    
    def _on_file_added(self, file_path: str):
        """Handle file addition event"""
        logger.info(f"File addition event: {file_path}")
        for callback in self._file_added_callbacks:
            try:
                callback(file_path)
            except Exception as e:
                logger.error(f"File addition callback execution failed: {str(e)}")
    
    def _on_file_removed(self, file_path: str):
        """Handle file deletion event"""
        logger.info(f"File deletion event: {file_path}")
        for callback in self._file_removed_callbacks:
            try:
                callback(file_path)
            except Exception as e:
                logger.error(f"File deletion callback execution failed: {str(e)}")
    
    def _on_file_modified(self, file_path: str):
        """Handle file modification event"""
        logger.info(f"File modification event: {file_path}")
        for callback in self._file_modified_callbacks:
            try:
                callback(file_path)
            except Exception as e:
                logger.error(f"File modification callback execution failed: {str(e)}")
    
    def start(self):
        """Start file system monitoring"""
        if self.is_running:
            logger.warning("File system watcher is already running")
            return
        
        # Ensure watch directory exists
        watch_path = Path(self.watch_directory)
        if not watch_path.exists():
            logger.error(f"Watch directory does not exist: {self.watch_directory}")
            return
        
        # Create event handler
        self.event_handler = DocumentFileHandler(
            on_file_added=self._on_file_added,
            on_file_removed=self._on_file_removed,
            on_file_modified=self._on_file_modified
        )
        
        # Create observer
        self.observer = Observer()
        self.observer.schedule(
            self.event_handler,
            self.watch_directory,
            recursive=True
        )
        
        # Start monitoring
        self.observer.start()
        self.is_running = True
        
        logger.info(f"File system watcher started, watching directory: {self.watch_directory}")
    
    def stop(self):
        """Stop file system monitoring"""
        if not self.is_running:
            logger.warning("File system watcher is not running")
            return
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        self.event_handler = None
        self.is_running = False
        
        logger.info("File system watcher stopped")
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop() 