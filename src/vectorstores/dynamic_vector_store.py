import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Callable
import threading
from datetime import datetime

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .vector_store import VectorStoreManager
from .file_watcher import FileSystemWatcher
from ..mcp.mcp_client import MCPFilesystemManager
from ..document_loaders.document_loader import DocumentLoaderManager
from ..config.settings import settings

logger = logging.getLogger(__name__)


class DynamicVectorStoreManager(VectorStoreManager):
    """动态FAISS向量存储管理器"""
    
    def __init__(self, use_openai_embeddings: bool = False, 
                 enable_file_watching: bool = True,
                 enable_mcp: bool = True):
        """
        初始化动态向量存储管理器
        
        Args:
            use_openai_embeddings: 是否使用OpenAI embeddings
            enable_file_watching: 是否启用文件监控
            enable_mcp: 是否启用MCP
        """
        super().__init__(use_openai_embeddings)
        
        # 动态管理相关属性
        self.enable_file_watching = enable_file_watching
        self.enable_mcp = enable_mcp
        
        # 文件监控器
        self.file_watcher: Optional[FileSystemWatcher] = None
        
        # MCP管理器
        self.mcp_manager: Optional[MCPFilesystemManager] = None
        if self.enable_mcp:
            self.mcp_manager = MCPFilesystemManager()
        
        # 文件状态跟踪
        self._file_document_mapping: Dict[str, List[str]] = {}  # 文件路径 -> 文档ID列表
        self._processing_files: Set[str] = set()  # 正在处理的文件
        self._file_last_modified: Dict[str, float] = {}  # 文件最后修改时间
        
        # 线程锁
        self._lock = threading.RLock()
        
        logger.info(f"动态向量存储管理器初始化 - 文件监控: {enable_file_watching}, MCP: {enable_mcp}")
    
    async def initialize(self, store_name: str = "default", force_recreate: bool = False) -> FAISS:
        """初始化动态向量存储"""
        logger.info("初始化动态向量存储...")
        
        # 初始化MCP
        if self.mcp_manager:
            await self.mcp_manager.initialize()
        
        # 检查是否需要自动重建向量存储
        if settings.auto_rebuild_vector_store:
            logger.info("🔄 配置了自动重建向量存储，正在重建...")
            force_recreate = True
        
        # 创建或加载向量存储 - 使用动态模式的逻辑
        vector_store = await self._get_or_create_dynamic_vector_store(store_name, force_recreate)
        
        # 建立文件到文档的映射
        await self._build_file_document_mapping()
        
        # 启动文件监控
        if self.enable_file_watching:
            await self._start_file_watching()
        
        logger.info("动态向量存储初始化完成")
        return vector_store
    
    async def _get_or_create_dynamic_vector_store(self, store_name: str = "default", force_recreate: bool = False) -> FAISS:
        """获取或创建动态向量存储 - 使用 DOCUMENT_DIRECTORIES 配置"""
        if force_recreate or not self._vector_store_exists(store_name):
            logger.info("创建新的动态向量存储...")
            await self._create_dynamic_vector_store()
            self.save_vector_store(store_name)
        else:
            logger.info("加载现有向量存储...")
            self.load_vector_store(store_name)
        
        return self.vector_store
    
    async def _create_dynamic_vector_store(self) -> FAISS:
        """创建动态向量存储 - 从 DOCUMENT_DIRECTORIES 加载文档"""
        logger.info("从配置的文档目录加载文档...")
        
        all_documents = []
        
        # 遍历所有配置的文档目录
        for directory_path in settings.document_directories:
            data_dir = Path(directory_path)
            
            if data_dir.exists():
                logger.info(f"正在加载目录: {directory_path}")
                try:
                    documents = self.document_loader.load_documents_from_directory(str(data_dir))
                    all_documents.extend(documents)
                    logger.info(f"从 {directory_path} 加载了 {len(documents)} 个文档片段")
                except Exception as e:
                    logger.error(f"加载目录失败 {directory_path}: {str(e)}")
            else:
                logger.warning(f"目录不存在，跳过: {directory_path}")
        
        if not all_documents:
            raise ValueError(f"未在配置的目录中找到可用的文档: {settings.document_directories}")
        
        logger.info(f"总共加载 {len(all_documents)} 个文档片段用于创建向量存储")
        logger.info(f"使用的文档目录: {settings.document_directories}")
        
        try:
            self.vector_store = FAISS.from_documents(
                documents=all_documents,
                embedding=self.embeddings
            )
            logger.info("动态向量存储创建成功")
            return self.vector_store
        except Exception as e:
            logger.error(f"创建动态向量存储失败: {str(e)}")
            raise
    
    async def cleanup(self):
        """清理资源"""
        logger.info("清理动态向量存储资源...")
        
        # 停止文件监控
        if self.file_watcher:
            self.file_watcher.stop()
            self.file_watcher = None
        
        # 清理MCP连接
        if self.mcp_manager:
            await self.mcp_manager.cleanup()
        
        logger.info("动态向量存储资源清理完成")
    
    async def _build_file_document_mapping(self):
        """建立文件到文档的映射关系"""
        logger.info("建立文件到文档的映射关系...")
        
        if not self.vector_store:
            logger.warning("向量存储未初始化，无法建立映射")
            return
        
        with self._lock:
            self._file_document_mapping.clear()
            self._file_last_modified.clear()
            
            # 遍历所有文档，建立映射
            for doc_id, document in self.vector_store.docstore._dict.items():
                if hasattr(document, 'metadata') and 'source' in document.metadata:
                    source_path = document.metadata['source']
                    
                    if source_path not in self._file_document_mapping:
                        self._file_document_mapping[source_path] = []
                    
                    self._file_document_mapping[source_path].append(doc_id)
                    
                    # 记录文件最后修改时间
                    try:
                        file_path = Path(source_path)
                        if file_path.exists():
                            self._file_last_modified[source_path] = file_path.stat().st_mtime
                    except Exception as e:
                        logger.warning(f"无法获取文件修改时间 {source_path}: {str(e)}")
        
        logger.info(f"建立了 {len(self._file_document_mapping)} 个文件的映射关系")
    
    async def _start_file_watching(self):
        """启动文件监控"""
        if not self.enable_file_watching:
            return
        
        logger.info("启动文件系统监控...")
        
        # 创建文件监控器 - 使用配置的监控目录
        self.file_watcher = FileSystemWatcher(settings.vector_watch_directory)
        
        # 注册回调函数
        self.file_watcher.add_file_added_callback(self._on_file_added)
        self.file_watcher.add_file_removed_callback(self._on_file_removed)
        self.file_watcher.add_file_modified_callback(self._on_file_modified)
        
        # 启动监控
        self.file_watcher.start()
        
        logger.info(f"文件系统监控已启动，监控目录: {settings.vector_watch_directory}")
    
    def _on_file_added(self, file_path: str):
        """处理文件添加事件"""
        logger.info(f"文件添加事件: {file_path}")
        
        # 使用线程池异步处理，避免阻塞文件监控
        self._schedule_async_task(self._handle_file_added(file_path))
    
    def _on_file_removed(self, file_path: str):
        """处理文件删除事件"""
        logger.info(f"文件删除事件: {file_path}")
        
        # 使用线程池异步处理
        self._schedule_async_task(self._handle_file_removed(file_path))
    
    def _on_file_modified(self, file_path: str):
        """处理文件修改事件"""
        logger.info(f"文件修改事件: {file_path}")
        
        # 使用线程池异步处理
        self._schedule_async_task(self._handle_file_modified(file_path))
    
    def _schedule_async_task(self, coro):
        """安全地调度异步任务"""
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_running_loop()
            # 在主线程的事件循环中创建任务
            loop.create_task(coro)
        except RuntimeError:
            # 如果没有运行的事件循环，使用线程池执行器
            import threading
            import concurrent.futures
            
            def run_async_task():
                try:
                    asyncio.run(coro)
                except Exception as e:
                    logger.error(f"异步任务执行失败: {str(e)}")
            
            # 使用线程池执行异步任务
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(run_async_task)
    
    async def _handle_file_added(self, file_path: str):
        """异步处理文件添加"""
        try:
            # 防止重复处理
            if file_path in self._processing_files:
                logger.debug(f"文件正在处理中，跳过: {file_path}")
                return
            
            with self._lock:
                self._processing_files.add(file_path)
            
            logger.info(f"开始处理新添加的文件: {file_path}")
            
            # 加载文档
            documents = self.document_loader.load_single_file(file_path)
            
            if documents:
                # 添加到向量存储
                self.add_documents(documents)
                
                # 更新映射 - 使用实际的文档ID
                with self._lock:
                    # 尝试多种路径格式获取文档ID
                    normalized_paths = [
                        file_path,
                        str(Path(file_path)),
                        str(Path(file_path).as_posix()),
                        file_path.replace('./', ''),
                        str(Path(file_path).relative_to('.') if file_path.startswith('./') else Path(file_path))
                    ]
                    
                    doc_ids = []
                    for path_variant in normalized_paths:
                        doc_ids = self.get_document_ids_by_source(path_variant)
                        if doc_ids:
                            logger.debug(f"找到文档ID使用路径格式: {path_variant}")
                            break
                    
                    self._file_document_mapping[file_path] = doc_ids
                    self._file_last_modified[file_path] = Path(file_path).stat().st_mtime
                
                # 保存向量存储
                self.save_vector_store()
                
                logger.info(f"成功添加文件到向量存储: {file_path}, 文档数: {len(documents)}")
            else:
                logger.warning(f"文件加载失败或无内容: {file_path}")
                
        except Exception as e:
            logger.error(f"处理文件添加失败 {file_path}: {str(e)}")
        finally:
            with self._lock:
                self._processing_files.discard(file_path)
    
    async def _handle_file_removed(self, file_path: str):
        """异步处理文件删除"""
        try:
            logger.info(f"开始处理文件删除: {file_path}")
            
            # 规范化路径格式，尝试多种格式匹配
            normalized_paths = [
                file_path,
                str(Path(file_path)),
                str(Path(file_path).as_posix()),
                file_path.replace('./', ''),
                str(Path(file_path).relative_to('.') if file_path.startswith('./') else Path(file_path))
            ]
            
            success = False
            for path_variant in normalized_paths:
                try:
                    if self.delete_documents_by_source(path_variant):
                        success = True
                        logger.info(f"成功使用路径格式删除文档: {path_variant}")
                        break
                except Exception as e:
                    logger.debug(f"路径格式 {path_variant} 删除失败: {str(e)}")
                    continue
            
            if success:
                with self._lock:
                    # 更新映射 - 删除所有可能的路径格式
                    for path_variant in normalized_paths:
                        if path_variant in self._file_document_mapping:
                            del self._file_document_mapping[path_variant]
                        self._file_last_modified.pop(path_variant, None)
                
                # 保存向量存储
                self.save_vector_store()
                
                logger.info(f"成功从向量存储中删除文件: {file_path}")
            else:
                logger.warning(f"文件可能不存在于向量存储中: {file_path}")
                logger.debug(f"尝试的路径格式: {normalized_paths}")
                
        except Exception as e:
            logger.error(f"处理文件删除失败 {file_path}: {str(e)}")
    
    async def _handle_file_modified(self, file_path: str):
        """异步处理文件修改"""
        try:
            # 防止重复处理
            if file_path in self._processing_files:
                logger.debug(f"文件正在处理中，跳过: {file_path}")
                return
            
            # 检查文件是否真的被修改
            try:
                current_mtime = Path(file_path).stat().st_mtime
                last_mtime = self._file_last_modified.get(file_path, 0)
                
                if abs(current_mtime - last_mtime) < 1:  # 1秒内的修改认为是同一次
                    logger.debug(f"文件修改时间差异过小，跳过: {file_path}")
                    return
            except Exception:
                pass  # 如果获取修改时间失败，继续处理
            
            with self._lock:
                self._processing_files.add(file_path)
            
            logger.info(f"开始处理文件修改: {file_path}")
            
            # 先删除旧文档
            await self._handle_file_removed(file_path)
            
            # 再添加新文档
            await self._handle_file_added(file_path)
            
            logger.info(f"文件修改处理完成: {file_path}")
            
        except Exception as e:
            logger.error(f"处理文件修改失败 {file_path}: {str(e)}")
        finally:
            with self._lock:
                self._processing_files.discard(file_path)
    
    def get_file_document_mapping(self) -> Dict[str, List[str]]:
        """获取文件到文档的映射关系"""
        with self._lock:
            return self._file_document_mapping.copy()
    
    def get_processing_files(self) -> Set[str]:
        """获取正在处理的文件列表"""
        with self._lock:
            return self._processing_files.copy()
    
    async def force_sync_with_filesystem(self):
        """强制与文件系统同步"""
        logger.info("开始强制同步文件系统...")
        
        try:
            # 获取当前文件系统中的所有文件
            current_files = set()
            
            # 遍历所有配置的文档目录
            for directory_path in settings.document_directories:
                data_dir = Path(directory_path)
                
                if data_dir.exists():
                    logger.info(f"正在扫描目录: {directory_path}")
                    for file_path in data_dir.rglob("*"):
                        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.xls']:
                            current_files.add(str(file_path))
                else:
                    logger.warning(f"目录不存在，跳过: {directory_path}")
            
            # 获取向量存储中记录的文件
            with self._lock:
                stored_files = set(self._file_document_mapping.keys())
            
            # 找出需要添加的文件
            files_to_add = current_files - stored_files
            
            # 找出需要删除的文件
            files_to_remove = stored_files - current_files
            
            logger.info(f"同步统计 - 需要添加: {len(files_to_add)}, 需要删除: {len(files_to_remove)}")
            logger.info(f"扫描的目录: {settings.document_directories}")
            
            # 处理删除
            for file_path in files_to_remove:
                await self._handle_file_removed(file_path)
            
            # 处理添加
            for file_path in files_to_add:
                await self._handle_file_added(file_path)
            
            logger.info("文件系统同步完成")
            
        except Exception as e:
            logger.error(f"文件系统同步失败: {str(e)}")
    
    def get_status(self) -> Dict[str, any]:
        """获取动态向量存储状态"""
        with self._lock:
            return {
                "file_watching_enabled": self.enable_file_watching,
                "mcp_enabled": self.enable_mcp,
                "mcp_available": self.mcp_manager.is_available() if self.mcp_manager else False,
                "file_watcher_running": self.file_watcher.is_running if self.file_watcher else False,
                "tracked_files_count": len(self._file_document_mapping),
                "processing_files_count": len(self._processing_files),
                "tracked_files": list(self._file_document_mapping.keys()),
                "processing_files": list(self._processing_files),
                "last_sync_time": datetime.now().isoformat()
            } 