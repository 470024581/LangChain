import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..config.settings import settings

logger = logging.getLogger(__name__)


class MCPFilesystemClient:
    """LangChain MCP Filesystem Client"""
    
    def __init__(self, allowed_directories: List[str] = None):
        """Initialize LangChain MCP Filesystem Client"""
        self.allowed_directories = allowed_directories or [
            str(Path(settings.data_directory).absolute()),
            str(Path(settings.vector_store_path).absolute()),
            str(Path(".").absolute())
        ]
        self.is_connected = False
        self.mcp_client: Optional[Any] = None
        
        # Configure MCP filesystem server
        self.mcp_config = {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                "transport": "stdio",
                "cwd": str(Path(".").absolute())
            }
        }
        
        logger.info(f"LangChain MCP Filesystem client initialized, allowed directories: {self.allowed_directories}")
    
    async def connect(self) -> bool:
        """Connect to MCP server"""
        try:
            logger.info("Connecting to filesystem server using LangChain MCP adapter...")
            
            # Delayed import to avoid circular imports
            from langchain_mcp_adapters.client import MultiServerMCPClient
            
            # Create LangChain MCP client
            self.mcp_client = MultiServerMCPClient(self.mcp_config)
            
            # Test connection (by getting tool list)
            tools = await self.mcp_client.get_tools()
            logger.info(f"Successfully connected, found {len(tools)} MCP tools")
            
            self.is_connected = True
            logger.info("LangChain MCP Filesystem client connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"LangChain MCP Filesystem client connection failed: {str(e)}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect MCP connection"""
        if self.is_connected and self.mcp_client:
            try:
                # MultiServerMCPClient handles cleanup automatically
                logger.info("Disconnecting LangChain MCP Filesystem connection")
                self.is_connected = False
                self.mcp_client = None
            except Exception as e:
                logger.error(f"Error occurred during disconnection: {str(e)}")
    
    async def get_tools(self) -> List:
        """Get MCP tool list"""
        if not self.is_connected or not self.mcp_client:
            logger.warning("MCP client not connected")
            return []
        
        try:
            tools = await self.mcp_client.get_tools()
            logger.info(f"Retrieved {len(tools)} MCP tools")
            return tools
        except Exception as e:
            logger.error(f"Failed to get MCP tools: {str(e)}")
            return []
    
    def _is_allowed_directory(self, directory: Path) -> bool:
        """Check if directory is within allowed access range"""
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
    """LangChain MCP Filesystem Manager"""
    
    def __init__(self):
        """Initialize LangChain MCP Filesystem Manager"""
        self.client: Optional[MCPFilesystemClient] = None
        self.is_enabled = getattr(settings, 'mcp_enabled', False)
        self.filesystem_enabled = getattr(settings, 'mcp_filesystem_enabled', False)
        
        if self.is_enabled and self.filesystem_enabled:
            allowed_dirs = getattr(settings, 'mcp_filesystem_allowed_directories', [])
            self.client = MCPFilesystemClient(allowed_directories=allowed_dirs)
    
    async def initialize(self) -> bool:
        """Initialize MCP connection"""
        if not self.is_enabled or not self.filesystem_enabled:
            logger.info("LangChain MCP Filesystem not enabled")
            return False
        
        if self.client:
            return await self.client.connect()
        
        return False
    
    async def cleanup(self):
        """Clean up MCP connection"""
        if self.client:
            await self.client.disconnect()
    
    async def get_tools(self) -> List:
        """Get MCP tools"""
        if self.client and self.client.is_connected:
            return await self.client.get_tools()
        return []
    
    def is_available(self) -> bool:
        """Check if LangChain MCP Filesystem is available"""
        return (self.client is not None and 
                self.client.is_connected and 
                self.is_enabled and 
                self.filesystem_enabled)


# Create global MCP manager instance
mcp_manager = MCPFilesystemManager() 