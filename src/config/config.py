"""
Compatible configuration module providing nested access support
"""
from .settings import settings


class OpenRouterConfig:
    """OpenRouter configuration class"""
    
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
    """Configuration class providing nested access"""
    
    def __init__(self):
        self.openrouter = OpenRouterConfig()
    
    # Direct access to settings attributes
    def __getattr__(self, name):
        return getattr(settings, name)


# Global configuration instance
config = Config() 