import os
from typing import Optional, List
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # OpenRouter API configuration
    openrouter_api_key: Optional[str] = Field(None, env="OPENROUTER_API_KEY")
    openrouter_api_base: str = Field("https://openrouter.ai/api/v1", env="OPENROUTER_API_BASE")
    
    # Default model configuration
    default_model: str = Field("gpt-4o-mini", env="DEFAULT_MODEL")
    
    # LLM provider ("openrouter" or "ollama")
    llm_provider: str = Field("openrouter", env="LLM_PROVIDER")
    
    # Ollama configuration (if using ollama)
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("mistral", env="OLLAMA_MODEL")
    
    # LLM parameter configuration
    openai_temperature: float = Field(0.1, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(2000, env="OPENAI_MAX_TOKENS")
    openai_top_p: float = Field(1.0, env="OPENAI_TOP_P")
    openai_frequency_penalty: float = Field(0.0, env="OPENAI_FREQUENCY_PENALTY")
    openai_presence_penalty: float = Field(0.0, env="OPENAI_PRESENCE_PENALTY")
    
    # Vector database configuration
    vector_store_path: str = Field("./vector_store", env="VECTOR_STORE_PATH")
    
    # API service configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    
    # Document processing configuration
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # Data directory configuration
    data_directory: str = Field("./data", env="DATA_DIRECTORY")
    # Document directory configuration (internal field for parsing)
    document_directories_str: str = Field("./data/document", env="DOCUMENT_DIRECTORIES")
    # MCP directory configuration (internal field for parsing)
    mcp_directories_str: str = Field("./data,./vector_store,./", env="MCP_FILESYSTEM_ALLOWED_DIRECTORIES")
    # Vector store monitoring directory (for file monitoring)
    vector_watch_directory: str = Field("./data", env="VECTOR_WATCH_DIRECTORY")
    
    # Retrieval configuration
    retriever_k: int = Field(5, env="RETRIEVER_K")
    search_type: str = Field("similarity", env="SEARCH_TYPE")
    
    # Memory configuration
    memory_type: str = Field("buffer_window", env="MEMORY_TYPE")
    memory_k: int = Field(5, env="MEMORY_K")
    max_token_limit: int = Field(2000, env="MAX_TOKEN_LIMIT")
    
    # LangSmith configuration
    langchain_tracing_v2: bool = Field(False, env="LANGCHAIN_TRACING_V2")
    langsmith_api_key: Optional[str] = Field(None, env="LANGSMITH_API_KEY")
    langchain_project: str = Field("Default", env="LANGCHAIN_PROJECT")
    langchain_endpoint: str = Field("https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT")
    
    # Embedding model
    embedding_model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL_NAME")
    
    # Reranker model
    reranker_model: str = Field("BAAI/bge-reranker-base", env="RERANKER_MODEL")
    
    # MCP configuration
    mcp_enabled: bool = Field(True, env="MCP_ENABLED")
    mcp_filesystem_enabled: bool = Field(True, env="MCP_FILESYSTEM_ENABLED")
    
    # Dynamic vector store configuration - enabled by default
    enable_dynamic_vector_store: bool = Field(True, env="ENABLE_DYNAMIC_VECTOR_STORE")
    enable_file_watching: bool = Field(True, env="ENABLE_FILE_WATCHING")
    auto_sync_filesystem: bool = Field(True, env="AUTO_SYNC_FILESYSTEM")
    file_watch_delay: float = Field(1.0, env="FILE_WATCH_DELAY")
    
    # Auto rebuild vector store on startup
    auto_rebuild_vector_store: bool = Field(False, env="AUTO_REBUILD_VECTOR_STORE")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # Ignore extra fields for better compatibility
    }
    
    @computed_field
    @property
    def document_directories(self) -> List[str]:
        """Get document directory list"""
        return [dir.strip() for dir in self.document_directories_str.split(',') if dir.strip()]
    
    @computed_field
    @property
    def mcp_filesystem_allowed_directories(self) -> List[str]:
        """Get MCP allowed directory list"""
        return [dir.strip() for dir in self.mcp_directories_str.split(',') if dir.strip()]


# Global configuration instance
settings = Settings() 