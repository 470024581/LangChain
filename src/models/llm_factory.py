import logging
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain_ollama import ChatOllama

from ..config.settings import settings

logger = logging.getLogger(__name__)


class LLMFactory:
    """大语言模型工厂类"""
    
    @staticmethod
    def create_ollama_llm(
        model_name: str = None,
        temperature: float = 0.1,
        **kwargs
    ) -> ChatOllama:
        """
        创建Ollama LLM实例
        
        Args:
            model_name: 模型名称，默认使用配置中的模型
            temperature: 温度参数
            **kwargs: 其他参数
        
        Returns:
            ChatOllama: LLM实例
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
            logger.info(f"成功创建Ollama LLM: {model_name} at {settings.ollama_base_url}")
            return llm
        except Exception as e:
            logger.error(f"创建Ollama LLM失败: {str(e)}")
            raise
    
    @staticmethod
    def create_openrouter_llm(
        model_name: str = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs
    ) -> ChatOpenAI:
        """
        创建OpenRouter LLM实例
        
        Args:
            model_name: 模型名称，默认使用配置中的模型
            temperature: 温度参数，控制输出随机性
            max_tokens: 最大输出token数
            **kwargs: 其他参数
        
        Returns:
            ChatOpenAI: LLM实例
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
            
            logger.info(f"成功创建OpenRouter LLM: {model_name}")
            return llm
            
        except Exception as e:
            logger.error(f"创建OpenRouter LLM失败: {str(e)}")
            raise
    
    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """获取可用的模型列表"""
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
        根据配置创建LLM实例，支持重试
        
        Args:
            model_name: 模型名称
            max_retries: 最大重试次数
            **kwargs: 其他参数
            
        Returns:
            BaseLanguageModel: LLM实例
        """
        provider = settings.llm_provider.lower()
        logger.info(f"使用LLM提供商: {provider}")

        for attempt in range(max_retries):
            try:
                if provider == "ollama":
                    return LLMFactory.create_ollama_llm(model_name, **kwargs)
                elif provider == "openrouter":
                    return LLMFactory.create_openrouter_llm(model_name, **kwargs)
                else:
                    raise ValueError(f"不支持的LLM提供商: {settings.llm_provider}")
            except Exception as e:
                logger.warning(f"创建LLM失败，尝试 {attempt + 1}/{max_retries}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
        
        raise RuntimeError("创建LLM失败，已达到最大重试次数")


class LLMManager:
    """LLM管理器，提供单例模式的LLM访问"""
    
    _instance: Optional['LLMManager'] = None
    _llm: Optional[BaseLanguageModel] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_llm(self, model_name: str = None, **kwargs) -> BaseLanguageModel:
        """获取LLM实例（单例模式）"""
        # 强制刷新LLM如果提供商或模型名称改变
        provider_changed = getattr(self._llm, '_provider', None) != settings.llm_provider.lower()
        model_changed = model_name and model_name != getattr(self._llm, 'model', 'unknown')

        if self._llm is None or provider_changed or model_changed:
            self._llm = LLMFactory.create_llm(model_name, **kwargs)
            # 附加提供商信息以便于检查
            self._llm._provider = settings.llm_provider.lower()
        return self._llm
    
    def refresh_llm(self, model_name: str = None, **kwargs) -> BaseLanguageModel:
        """刷新LLM实例"""
        self._llm = LLMFactory.create_llm(model_name, **kwargs)
        self._llm._provider = settings.llm_provider.lower()
        return self._llm 