import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..config.settings import settings

logger = logging.getLogger(__name__)


class MCPFilesystemClient:
    """LangChain MCP Filesystem客户端"""
    
    def __init__(self, allowed_directories: List[str] = None):
        """初始化LangChain MCP Filesystem客户端"""
        self.allowed_directories = allowed_directories or [
            str(Path(settings.data_directory).absolute()),
            str(Path(settings.vector_store_path).absolute()),
            str(Path(".").absolute())
        ]
        self.is_connected = False
        self.mcp_client: Optional[Any] = None
        
        # 配置MCP文件系统服务器
        self.mcp_config = {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                "transport": "stdio",
                "cwd": str(Path(".").absolute())
            }
        }
        
        logger.info(f"LangChain MCP Filesystem客户端初始化，允许访问目录: {self.allowed_directories}")
    
    async def connect(self) -> bool:
        """连接到MCP服务器"""
        try:
            logger.info("正在使用LangChain MCP适配器连接到文件系统服务器...")
            
            # 延迟导入避免循环导入
            from langchain_mcp_adapters.client import MultiServerMCPClient
            
            # 创建LangChain MCP客户端
            self.mcp_client = MultiServerMCPClient(self.mcp_config)
            
            # 测试连接（通过获取工具列表）
            tools = await self.mcp_client.get_tools()
            logger.info(f"成功连接，找到 {len(tools)} 个MCP工具")
            
            self.is_connected = True
            logger.info("LangChain MCP Filesystem客户端连接成功")
            return True
            
        except Exception as e:
            logger.error(f"LangChain MCP Filesystem客户端连接失败: {str(e)}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """断开MCP连接"""
        if self.is_connected and self.mcp_client:
            try:
                # MultiServerMCPClient会自动处理清理
                logger.info("断开LangChain MCP Filesystem连接")
                self.is_connected = False
                self.mcp_client = None
            except Exception as e:
                logger.error(f"断开连接时发生错误: {str(e)}")
    
    async def get_tools(self) -> List:
        """获取MCP工具列表"""
        if not self.is_connected or not self.mcp_client:
            logger.warning("MCP客户端未连接")
            return []
        
        try:
            tools = await self.mcp_client.get_tools()
            logger.info(f"获取到 {len(tools)} 个MCP工具")
            return tools
        except Exception as e:
            logger.error(f"获取MCP工具失败: {str(e)}")
            return []
    
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
    """LangChain MCP Filesystem管理器"""
    
    def __init__(self):
        """初始化LangChain MCP Filesystem管理器"""
        self.client: Optional[MCPFilesystemClient] = None
        self.is_enabled = getattr(settings, 'mcp_enabled', False)
        self.filesystem_enabled = getattr(settings, 'mcp_filesystem_enabled', False)
        
        if self.is_enabled and self.filesystem_enabled:
            allowed_dirs = getattr(settings, 'mcp_filesystem_allowed_directories', [])
            self.client = MCPFilesystemClient(allowed_directories=allowed_dirs)
    
    async def initialize(self) -> bool:
        """初始化MCP连接"""
        if not self.is_enabled or not self.filesystem_enabled:
            logger.info("LangChain MCP Filesystem未启用")
            return False
        
        if self.client:
            return await self.client.connect()
        
        return False
    
    async def cleanup(self):
        """清理MCP连接"""
        if self.client:
            await self.client.disconnect()
    
    async def get_tools(self) -> List:
        """获取MCP工具"""
        if self.client and self.client.is_connected:
            return await self.client.get_tools()
        return []
    
    def is_available(self) -> bool:
        """检查LangChain MCP Filesystem是否可用"""
        return (self.client is not None and 
                self.client.is_connected and 
                self.is_enabled and 
                self.filesystem_enabled)


# 创建全局MCP管理器实例
mcp_manager = MCPFilesystemManager() 