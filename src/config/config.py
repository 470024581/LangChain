"""
兼容的配置模块，提供嵌套访问支持
"""
from .settings import settings


class OpenRouterConfig:
    """OpenRouter配置类"""
    
    @property
    def model_name(self):
        return settings.default_model
    
    @property
    def api_key(self):
        return settings.openrouter_api_key
    
    @property
    def api_base(self):
        return settings.openrouter_api_base


class Config:
    """配置类，提供嵌套访问"""
    
    def __init__(self):
        self.openrouter = OpenRouterConfig()
    
    # 直接访问settings的属性
    def __getattr__(self, name):
        return getattr(settings, name)


# 全局配置实例
config = Config() 