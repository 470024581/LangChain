import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Callable
import threading
from datetime import datetime
import hashlib

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .vector_store import VectorStoreManager
from .file_watcher import FileSystemWatcher
from ..mcp_integration.mcp_client import MCPFilesystemManager
from ..document_loaders.document_loader import DocumentLoaderManager
from ..config.settings import settings

logger = logging.getLogger(__name__)


class DynamicVectorStoreManager(VectorStoreManager):
    """Dynamic FAISS Vector Store Manager"""
    
    def __init__(self, use_openai_embeddings: bool = False, 
                 enable_file_watching: bool = True,
                 enable_mcp: bool = True):
        """
        Initialize dynamic vector store manager
        
        Args:
            use_openai_embeddings: Whether to use OpenAI embeddings
            enable_file_watching: Whether to enable file monitoring
            enable_mcp: Whether to enable MCP
        """
        super().__init__(use_openai_embeddings)
        
        # Dynamic management related attributes
        self.enable_file_watching = enable_file_watching
        self.enable_mcp = enable_mcp
        
        # File watcher
        self.file_watcher: Optional[FileSystemWatcher] = None
        
        # MCP manager
        self.mcp_manager: Optional[MCPFilesystemManager] = None
        if self.enable_mcp:
            self.mcp_manager = MCPFilesystemManager()
        
        # File state tracking
        self._file_document_mapping: Dict[str, List[str]] = {}  # File path -> Document ID list
        self._processing_files: Set[str] = set()  # Files being processed
        self._file_last_modified: Dict[str, float] = {}  # File last modified time
        self._file_hashes: Dict[str, str] = {} # To store file content hashes
        
        # Thread lock
        self._lock = threading.RLock()
        
        logger.info(f"Dynamic vector store manager initialized - File monitoring: {enable_file_watching}, MCP: {enable_mcp}")
    
    async def initialize(self, store_name: str = "default", force_recreate: bool = False) -> FAISS:
        """Initialize dynamic vector store"""
        logger.info("Initializing dynamic vector store...")
        
        # Initialize LangChain MCP
        if self.mcp_manager:
            success = await self.mcp_manager.initialize()
            if success:
                logger.info("LangChain MCP filesystem service initialization successful")
                # Get available MCP tools
                tools = await self.mcp_manager.get_tools()
                logger.info(f"Available LangChain MCP tools: {[tool.name for tool in tools]}")
            else:
                logger.warning("LangChain MCP filesystem service initialization failed")
        
        # Check if auto rebuild vector store is needed
        if settings.auto_rebuild_vector_store:
            logger.info("🔄 Auto rebuild vector store configured, rebuilding...")
            force_recreate = True
        
        # Create or load vector store - using dynamic mode logic
        vector_store = await self._get_or_create_dynamic_vector_store(store_name, force_recreate)
        
        # Build file to document mapping
        await self._build_file_document_mapping()
        
        # Start file monitoring
        if self.enable_file_watching:
            await self._start_file_watching()
        
        logger.info("Dynamic vector store initialization completed")
        return vector_store
    
    async def _get_or_create_dynamic_vector_store(self, store_name: str = "default", force_recreate: bool = False) -> FAISS:
        """Get or create dynamic vector store - using DOCUMENT_DIRECTORIES configuration"""
        if force_recreate or not self._vector_store_exists(store_name):
            logger.info("Creating new dynamic vector store...")
            await self._create_dynamic_vector_store()
            self.save_vector_store(store_name)
        else:
            logger.info("Loading existing vector store...")
            self.load_vector_store(store_name)
        
        return self.vector_store
    
    async def _create_dynamic_vector_store(self) -> FAISS:
        """Create dynamic vector store - load documents from DOCUMENT_DIRECTORIES"""
        logger.info("Loading documents from configured document directories...")
        
        all_documents = []
        
        # Iterate through all configured document directories
        for directory_path in settings.document_directories:
            data_dir = Path(directory_path)
            
            if data_dir.exists():
                logger.info(f"Loading directory: {directory_path}")
                try:
                    documents = self.document_loader.load_documents_from_directory(str(data_dir))
                    all_documents.extend(documents)
                    logger.info(f"Loaded {len(documents)} document chunks from {directory_path}")
                except Exception as e:
                    logger.error(f"Failed to load directory {directory_path}: {str(e)}")
            else:
                logger.warning(f"Directory does not exist, skipping: {directory_path}")
        
        if not all_documents:
            raise ValueError(f"No available documents found in configured directories: {settings.document_directories}")
        
        logger.info(f"Total loaded {len(all_documents)} document chunks for creating vector store")
        logger.info(f"Used document directories: {settings.document_directories}")
        
        try:
            self.vector_store = FAISS.from_documents(
                documents=all_documents,
                embedding=self.embeddings
            )
            logger.info("Dynamic vector store created successfully")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to create dynamic vector store: {str(e)}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up dynamic vector store resources...")
        
        # Stop file monitoring
        if self.file_watcher:
            self.file_watcher.stop()
            self.file_watcher = None
        
        # Clean up MCP connections
        if self.mcp_manager:
            await self.mcp_manager.cleanup()
        
        logger.info("Dynamic vector store resource cleanup completed")
    
    async def _build_file_document_mapping(self):
        """Build file to document mapping relationship"""
        logger.info("Building file to document mapping relationship...")
        
        if not self.vector_store:
            logger.warning("Vector store not initialized, cannot build mapping")
            return
        
        with self._lock:
            self._file_document_mapping.clear()
            self._file_last_modified.clear()
            
            # Iterate through all documents to build mapping
            for doc_id, document in self.vector_store.docstore._dict.items():
                if hasattr(document, 'metadata') and 'source' in document.metadata:
                    source_path = document.metadata['source']
                    
                    if source_path not in self._file_document_mapping:
                        self._file_document_mapping[source_path] = []
                    
                    self._file_document_mapping[source_path].append(doc_id)
                    
                    # Record file last modification time
                    try:
                        file_path = Path(source_path)
                        if file_path.exists():
                            self._file_last_modified[source_path] = file_path.stat().st_mtime
                    except Exception as e:
                        logger.warning(f"Unable to get file modification time {source_path}: {str(e)}")
        
        # After building the mapping, also build the initial hashes
        logger.info("Calculating initial file hashes...")
        for file_path in self._file_document_mapping.keys():
            self._file_hashes[file_path] = self._calculate_file_hash(file_path)
        logger.info(f"Initial hashes calculated for {len(self._file_hashes)} files.")
        
        logger.info(f"Built mapping relationship for {len(self._file_document_mapping)} files")
    
    async def _start_file_watching(self):
        """Start file monitoring"""
        if not self.enable_file_watching:
            return
        
        logger.info("Starting file system monitoring...")
        
        # Create file watcher - use configured monitoring directory
        self.file_watcher = FileSystemWatcher(settings.vector_watch_directory)
        
        # Register callback functions
        self.file_watcher.add_file_added_callback(self._on_file_added)
        self.file_watcher.add_file_removed_callback(self._on_file_removed)
        self.file_watcher.add_file_modified_callback(self._on_file_modified)
        
        # Start monitoring
        self.file_watcher.start()
        
        logger.info(f"File system monitoring started, monitoring directory: {settings.vector_watch_directory}")
    
    def _on_file_added(self, file_path: str):
        """Handle file addition event"""
        logger.info(f"File addition event: {file_path}")
        
        # Use thread pool for async processing to avoid blocking file monitoring
        self._schedule_async_task(self._handle_file_added(file_path))
    
    def _on_file_removed(self, file_path: str):
        """Handle file deletion event"""
        logger.info(f"File deletion event: {file_path}")
        
        # Use thread pool for async processing
        self._schedule_async_task(self._handle_file_removed(file_path))
    
    def _on_file_modified(self, file_path: str):
        """Handle file modification event"""
        logger.info(f"File modification event: {file_path}")
        
        # Use thread pool for async processing
        self._schedule_async_task(self._handle_file_modified(file_path))
    
    def _schedule_async_task(self, coro):
        """Safely schedule async task"""
        try:
            # Try to get current event loop
            loop = asyncio.get_running_loop()
            # Create task in main thread's event loop
            loop.create_task(coro)
        except RuntimeError:
            # If no running event loop, use thread pool executor
            import threading
            import concurrent.futures
            
            def run_async_task():
                try:
                    asyncio.run(coro)
                except Exception as e:
                    logger.error(f"Async task execution failed: {str(e)}")
            
            # Use thread pool to execute async task
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(run_async_task)
    
    async def _handle_file_added(self, file_path: str):
        """Async handle file addition"""
        try:
            # Prevent duplicate processing
            if file_path in self._processing_files:
                logger.debug(f"File is being processed, skipping: {file_path}")
                return
            
            with self._lock:
                self._processing_files.add(file_path)
            
            logger.info(f"Starting to process newly added file: {file_path}")
            
            # Load documents
            documents = self.document_loader.load_single_file(file_path)
            
            if documents:
                # Add to vector store
                self.add_documents(documents)
                
                # Update mapping - use actual document IDs
                with self._lock:
                    # Try multiple path formats to get document IDs
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
                            logger.debug(f"Found document IDs using path format: {path_variant}")
                            break
                    
                    self._file_document_mapping[file_path] = doc_ids
                    self._file_last_modified[file_path] = Path(file_path).stat().st_mtime
                
                # Save vector store
                self.save_vector_store()
                
                # At the end of the function, store the hash
                new_hash = self._calculate_file_hash(file_path)
                if new_hash:
                    self._file_hashes[file_path] = new_hash
                
                logger.info(f"Successfully added file to vector store: {file_path}, document count: {len(documents)}")
            else:
                logger.warning(f"File loading failed or no content: {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to process file addition {file_path}: {str(e)}")
        finally:
            with self._lock:
                self._processing_files.discard(file_path)
    
    async def _handle_file_removed(self, file_path: str):
        """Async handle file removal"""
        try:
            logger.info(f"Starting to process file removal: {file_path}")
            
            # Normalize path format, try multiple format matching
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
                        logger.info(f"Successfully deleted documents using path format: {path_variant}")
                        break
                except Exception as e:
                    logger.debug(f"Path format {path_variant} deletion failed: {str(e)}")
                    continue
            
            if success:
                with self._lock:
                    # Update mapping - remove all possible path formats
                    for path_variant in normalized_paths:
                        if path_variant in self._file_document_mapping:
                            del self._file_document_mapping[path_variant]
                        self._file_last_modified.pop(path_variant, None)
                
                # Save vector store
                self.save_vector_store()
                
                # When removing, also clear the hash
                if file_path in self._file_hashes:
                    del self._file_hashes[file_path]
                
                logger.info(f"Successfully removed file from vector store: {file_path}")
            else:
                logger.warning(f"File may not exist in vector store: {file_path}")
                logger.debug(f"Attempted path formats: {normalized_paths}")
                
        except Exception as e:
            logger.error(f"Failed to process file removal {file_path}: {str(e)}")
    
    async def _handle_file_modified(self, file_path: str):
        """
        Handles file modification events with a content hash check to prevent unnecessary updates.
        """
        current_hash = self._calculate_file_hash(file_path)
        last_hash = self._file_hashes.get(file_path)

        if current_hash and current_hash == last_hash:
            logger.info(f"File content unchanged for {file_path}, skipping modification event.")
            return

        logger.info(f"File content has changed for {file_path}. Processing modification...")
        
        # Proceed with remove and add logic
        await self._handle_file_removed(file_path)
        await self._handle_file_added(file_path)
        
        # Update the hash
        if current_hash:
            self._file_hashes[file_path] = current_hash

    def get_file_document_mapping(self) -> Dict[str, List[str]]:
        """Get file to document mapping relationship"""
        with self._lock:
            return self._file_document_mapping.copy()
    
    def get_processing_files(self) -> Set[str]:
        """Get list of files being processed"""
        with self._lock:
            return self._processing_files.copy()
    
    async def force_sync_with_filesystem(self):
        """Force synchronization with filesystem"""
        logger.info("Starting forced filesystem synchronization...")
        
        try:
            # Get all files in current filesystem
            current_files = set()
            
            # Iterate through all configured document directories
            for directory_path in settings.document_directories:
                data_dir = Path(directory_path)
                
                if data_dir.exists():
                    logger.info(f"Scanning directory: {directory_path}")
                    for file_path in data_dir.rglob("*"):
                        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.xls']:
                            current_files.add(str(file_path))
                else:
                    logger.warning(f"Directory does not exist, skipping: {directory_path}")
            
            # Get files recorded in vector store
            with self._lock:
                stored_files = set(self._file_document_mapping.keys())
            
            # Find files to add
            files_to_add = current_files - stored_files
            
            # Find files to remove
            files_to_remove = stored_files - current_files
            
            logger.info(f"Sync statistics - to add: {len(files_to_add)}, to remove: {len(files_to_remove)}")
            logger.info(f"Scanned directories: {settings.document_directories}")
            
            # Process deletions
            for file_path in files_to_remove:
                await self._handle_file_removed(file_path)
            
            # Process additions
            for file_path in files_to_add:
                await self._handle_file_added(file_path)
            
            logger.info("Filesystem synchronization completed")
            
        except Exception as e:
            logger.error(f"Filesystem synchronization failed: {str(e)}")
    
    def get_status(self) -> Dict[str, any]:
        """Get dynamic vector store status"""
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
    
    def _calculate_file_hash(self, file_path: str) -> Optional[str]:
        """Calculates the SHA256 hash of a file's content."""
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
            return file_hash.hexdigest()
        except (IOError, OSError) as e:
            logger.warning(f"Could not calculate hash for file {file_path}: {e}")
            return None 