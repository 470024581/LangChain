import logging
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel

from ..config.settings import settings

logger = logging.getLogger(__name__)


class LLMFactory:
    """大语言模型工厂类"""
    
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
    def create_llm_with_retry(
        model_name: str = None,
        max_retries: int = 3,
        **kwargs
    ) -> ChatOpenAI:
        """
        创建带重试机制的LLM
        
        Args:
            model_name: 模型名称
            max_retries: 最大重试次数
            **kwargs: 其他参数
        
        Returns:
            ChatOpenAI: LLM实例
        """
        for attempt in range(max_retries):
            try:
                return LLMFactory.create_openrouter_llm(model_name, **kwargs)
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
        if self._llm is None or model_name != getattr(self._llm, 'model_name', None):
            self._llm = LLMFactory.create_openrouter_llm(model_name, **kwargs)
        return self._llm
    
    def refresh_llm(self, model_name: str = None, **kwargs) -> BaseLanguageModel:
        """刷新LLM实例"""
        self._llm = LLMFactory.create_openrouter_llm(model_name, **kwargs)
        return self._llm 