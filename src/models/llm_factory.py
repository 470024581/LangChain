import logging
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain_ollama import ChatOllama

from ..config.settings import settings

logger = logging.getLogger(__name__)


class LLMFactory:
    """Large Language Model factory class"""
    
    @staticmethod
    def create_ollama_llm(
        model_name: str = None,
        temperature: float = 0.1,
        **kwargs
    ) -> ChatOllama:
        """
        Create Ollama LLM instance
        
        Args:
            model_name: Model name, defaults to model in configuration
            temperature: Temperature parameter
            **kwargs: Other parameters
        
        Returns:
            ChatOllama: LLM instance
        """
        if model_name is None:
            model_name = settings.ollama_model
        
        try:
            llm = ChatOllama(
                model=model_name,
                base_url=settings.ollama_base_url,
                temperature=temperature,
                **kwargs
            )
            logger.info(f"Successfully created Ollama LLM: {model_name} at {settings.ollama_base_url}")
            return llm
        except Exception as e:
            logger.error(f"Failed to create Ollama LLM: {str(e)}")
            raise
    
    @staticmethod
    def create_openrouter_llm(
        model_name: str = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs
    ) -> ChatOpenAI:
        """
        Create OpenRouter LLM instance
        
        Args:
            model_name: Model name, defaults to model in configuration
            temperature: Temperature parameter, controls output randomness
            max_tokens: Maximum output token count
            **kwargs: Other parameters
        
        Returns:
            ChatOpenAI: LLM instance
        """
        if model_name is None:
            model_name = settings.default_model
        
        try:
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=settings.openrouter_api_key,
                openai_api_base=settings.openrouter_api_base,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            logger.info(f"Successfully created OpenRouter LLM: {model_name}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to create OpenRouter LLM: {str(e)}")
            raise
    
    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """Get list of available models"""
        return {
            "gpt-4o-mini": "OpenAI GPT-4o Mini",
            "gpt-4o": "OpenAI GPT-4o",
            "gpt-4-turbo": "OpenAI GPT-4 Turbo", 
            "claude-3-sonnet": "Anthropic Claude 3 Sonnet",
            "claude-3-haiku": "Anthropic Claude 3 Haiku",
            "llama-3.1-8b": "Meta Llama 3.1 8B",
            "llama-3.1-70b": "Meta Llama 3.1 70B"
        }
    
    @staticmethod
    def create_llm(
        model_name: str = None,
        max_retries: int = 3,
        **kwargs
    ) -> BaseLanguageModel:
        """
        Create LLM instance based on configuration, with retry support
        
        Args:
            model_name: Model name
            max_retries: Maximum retry count
            **kwargs: Other parameters
            
        Returns:
            BaseLanguageModel: LLM instance
        """
        provider = settings.llm_provider.lower()
        logger.info(f"Using LLM provider: {provider}")

        for attempt in range(max_retries):
            try:
                if provider == "ollama":
                    return LLMFactory.create_ollama_llm(model_name, **kwargs)
                elif provider == "openrouter":
                    return LLMFactory.create_openrouter_llm(model_name, **kwargs)
                else:
                    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
            except Exception as e:
                logger.warning(f"Failed to create LLM, attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
        
        raise RuntimeError("Failed to create LLM, maximum retry count reached")


class LLMManager:
    """LLM manager, providing singleton access to LLM"""
    
    _instance: Optional['LLMManager'] = None
    _llm: Optional[BaseLanguageModel] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_llm(self, model_name: str = None, **kwargs) -> BaseLanguageModel:
        """Get LLM instance (singleton pattern)"""
        # Force refresh LLM if provider or model name changes
        provider_changed = getattr(self._llm, '_provider', None) != settings.llm_provider.lower()
        model_changed = model_name and model_name != getattr(self._llm, 'model', 'unknown')

        if self._llm is None or provider_changed or model_changed:
            self._llm = LLMFactory.create_llm(model_name, **kwargs)
            # Attach provider information for checking
            self._llm._provider = settings.llm_provider.lower()
        return self._llm
    
    def refresh_llm(self, model_name: str = None, **kwargs) -> BaseLanguageModel:
        """Refresh LLM instance"""
        self._llm = LLMFactory.create_llm(model_name, **kwargs)
        self._llm._provider = settings.llm_provider.lower()
        return self._llm 