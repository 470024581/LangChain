import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..config.settings import settings

logger = logging.getLogger(__name__)


class MCPFilesystemClient:
    """MCP Filesystem客户端"""
    
    def __init__(self, allowed_directories: List[str] = None):
        """初始化MCP Filesystem客户端"""
        self.allowed_directories = allowed_directories or [
            str(Path(settings.data_directory).absolute()),
            str(Path(settings.vector_store_path).absolute()),
            str(Path(".").absolute())
        ]
        self.is_connected = False
        
        logger.info(f"MCP Filesystem客户端初始化，允许访问目录: {self.allowed_directories}")
    
    async def connect(self) -> bool:
        """连接到MCP服务器"""
        try:
            logger.info("正在连接到MCP Filesystem服务器...")
            await asyncio.sleep(0.1)  # 模拟连接过程
            
            self.is_connected = True
            logger.info("MCP Filesystem客户端连接成功")
            return True
            
        except Exception as e:
            logger.error(f"MCP Filesystem客户端连接失败: {str(e)}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """断开MCP连接"""
        if self.is_connected:
            logger.info("断开MCP Filesystem连接")
            self.is_connected = False
    
    def _is_allowed_directory(self, directory: Path) -> bool:
        """检查目录是否在允许访问的范围内"""
        directory_abs = directory.resolve()
        
        for allowed_dir in self.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                directory_abs.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        
        return False


class MCPFilesystemManager:
    """MCP Filesystem管理器"""
    
    def __init__(self):
        """初始化MCP Filesystem管理器"""
        self.client: Optional[MCPFilesystemClient] = None
        self.is_enabled = getattr(settings, 'mcp_enabled', False)
        self.filesystem_enabled = getattr(settings, 'mcp_filesystem_enabled', False)
        
        if self.is_enabled and self.filesystem_enabled:
            allowed_dirs = getattr(settings, 'mcp_filesystem_allowed_directories', [])
            self.client = MCPFilesystemClient(allowed_directories=allowed_dirs)
    
    async def initialize(self) -> bool:
        """初始化MCP连接"""
        if not self.is_enabled or not self.filesystem_enabled:
            logger.info("MCP Filesystem未启用")
            return False
        
        if self.client:
            return await self.client.connect()
        
        return False
    
    async def cleanup(self):
        """清理MCP连接"""
        if self.client:
            await self.client.disconnect()
    
    def is_available(self) -> bool:
        """检查MCP Filesystem是否可用"""
        return (self.client is not None and 
                self.client.is_connected and 
                self.is_enabled and 
                self.filesystem_enabled) 