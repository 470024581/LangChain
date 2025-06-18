import os
from typing import Optional, List
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用程序配置设置"""
    
    # OpenRouter API配置
    openrouter_api_key: Optional[str] = Field(None, env="OPENROUTER_API_KEY")
    openrouter_api_base: str = Field("https://openrouter.ai/api/v1", env="OPENROUTER_API_BASE")
    
    # 默认模型配置
    default_model: str = Field("gpt-4o-mini", env="DEFAULT_MODEL")
    
    # LLM提供商 ("openrouter" 或 "ollama")
    llm_provider: str = Field("openrouter", env="LLM_PROVIDER")
    
    # Ollama配置 (如果使用ollama)
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("mistral", env="OLLAMA_MODEL")
    
    # LLM参数配置
    openai_temperature: float = Field(0.1, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(2000, env="OPENAI_MAX_TOKENS")
    openai_top_p: float = Field(1.0, env="OPENAI_TOP_P")
    openai_frequency_penalty: float = Field(0.0, env="OPENAI_FREQUENCY_PENALTY")
    openai_presence_penalty: float = Field(0.0, env="OPENAI_PRESENCE_PENALTY")
    
    # 向量数据库配置
    vector_store_path: str = Field("./vector_store", env="VECTOR_STORE_PATH")
    
    # API服务配置
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    
    # 文档处理配置
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # 数据目录配置
    data_directory: str = Field("./data", env="DATA_DIRECTORY")
    # 文档目录配置（内部字段，用于解析）
    document_directories_str: str = Field("./data/document", env="DOCUMENT_DIRECTORIES")
    # MCP目录配置（内部字段，用于解析）
    mcp_directories_str: str = Field("./data,./vector_store,./", env="MCP_FILESYSTEM_ALLOWED_DIRECTORIES")
    # 向量存储监控目录（用于文件监控）
    vector_watch_directory: str = Field("./data", env="VECTOR_WATCH_DIRECTORY")
    
    # 检索配置
    retriever_k: int = Field(5, env="RETRIEVER_K")
    search_type: str = Field("similarity", env="SEARCH_TYPE")
    
    # 记忆配置
    memory_type: str = Field("buffer_window", env="MEMORY_TYPE")
    memory_k: int = Field(5, env="MEMORY_K")
    max_token_limit: int = Field(2000, env="MAX_TOKEN_LIMIT")
    
    # LangSmith配置
    langchain_tracing_v2: bool = Field(False, env="LANGCHAIN_TRACING_V2")
    langsmith_api_key: Optional[str] = Field(None, env="LANGSMITH_API_KEY")
    langchain_project: str = Field("Default", env="LANGCHAIN_PROJECT")
    langchain_endpoint: str = Field("https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT")
    
    # Embedding模型
    embedding_model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL_NAME")
    
    # Reranker模型
    reranker_model: str = Field("BAAI/bge-reranker-base", env="RERANKER_MODEL")
    
    # MCP 配置
    mcp_enabled: bool = Field(True, env="MCP_ENABLED")
    mcp_filesystem_enabled: bool = Field(True, env="MCP_FILESYSTEM_ENABLED")
    
    # 动态向量存储配置 - 默认启用
    enable_dynamic_vector_store: bool = Field(True, env="ENABLE_DYNAMIC_VECTOR_STORE")
    enable_file_watching: bool = Field(True, env="ENABLE_FILE_WATCHING")
    auto_sync_filesystem: bool = Field(True, env="AUTO_SYNC_FILESYSTEM")
    file_watch_delay: float = Field(1.0, env="FILE_WATCH_DELAY")
    
    # 启动时自动重建向量存储
    auto_rebuild_vector_store: bool = Field(False, env="AUTO_REBUILD_VECTOR_STORE")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # 忽略额外的字段，提高兼容性
    }
    
    @computed_field
    @property
    def document_directories(self) -> List[str]:
        """获取文档目录列表"""
        return [dir.strip() for dir in self.document_directories_str.split(',') if dir.strip()]
    
    @computed_field
    @property
    def mcp_filesystem_allowed_directories(self) -> List[str]:
        """获取MCP允许目录列表"""
        return [dir.strip() for dir in self.mcp_directories_str.split(',') if dir.strip()]


# 全局配置实例
settings = Settings() 