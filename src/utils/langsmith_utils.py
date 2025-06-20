"""
LangSmith integration utility module
"""

import os
import logging
from typing import Optional, Dict, Any
from functools import wraps

from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager

from ..config.settings import settings

logger = logging.getLogger(__name__)


class LangSmithManager:
    """LangSmith manager"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.tracer: Optional[LangChainTracer] = None
        self._is_enabled = False
        self._initialize()
    
    def _initialize(self):
        """Initialize LangSmith connection"""
        try:
            # Check if LangSmith API key exists
            if not settings.langsmith_api_key:
                logger.info("LangSmith API key not set, skipping LangSmith initialization")
                self._is_enabled = False
                return
                
            if settings.langchain_tracing_v2 and settings.langsmith_api_key:
                # Force set environment variables (temporary debugging)
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
                os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
                os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
                
                logger.info(f"Force set environment variable: PROJECT={settings.langchain_project}")
                
                # Initialize client
                self.client = Client(
                    api_url=settings.langchain_endpoint,
                    api_key=settings.langsmith_api_key
                )
                
                # Initialize tracer
                self.tracer = LangChainTracer(
                    project_name=settings.langchain_project,
                    client=self.client
                )
                
                self._is_enabled = True
                logger.info(f"LangSmith enabled, project: {settings.langchain_project}")
                
                # Test connection
                self._test_connection()
                
            else:
                logger.info("LangSmith not enabled")
                self._is_enabled = False
                
        except Exception as e:
            logger.warning(f"LangSmith initialization failed, will disable tracing: {str(e)}")
            self._is_enabled = False
    
    def _test_connection(self):
        """Test LangSmith connection"""
        try:
            if self.client:
                # Try to get project information
                projects = list(self.client.list_projects(limit=1))
                logger.info("LangSmith connection test successful")
        except Exception as e:
            logger.warning(f"LangSmith connection test failed: {str(e)}")
    
    @property
    def is_enabled(self) -> bool:
        """Check if LangSmith is enabled"""
        return self._is_enabled
    
    def get_callback_manager(self) -> Optional[CallbackManager]:
        """Get callback manager"""
        if self.is_enabled and self.tracer:
            return CallbackManager([self.tracer])
        return None
    
    def get_callbacks(self) -> list:
        """Get callbacks list"""
        if self.is_enabled and self.tracer:
            return [self.tracer]
        return []
    
    def create_run(
        self, 
        name: str, 
        inputs: Dict[str, Any],
        run_type: str = "chain",
        **kwargs
    ):
        """Create run record"""
        if not self.is_enabled or not self.client:
            return None
        
        try:
            return self.client.create_run(
                name=name,
                inputs=inputs,
                run_type=run_type,
                project_name=settings.langchain_project,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to create LangSmith run record: {str(e)}")
            return None
    
    def update_run(self, run_id: str, outputs: Dict[str, Any], **kwargs):
        """Update run record"""
        if not self.is_enabled or not self.client:
            return
        
        try:
            self.client.update_run(
                run_id=run_id,
                outputs=outputs,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to update LangSmith run record: {str(e)}")
    
    def log_feedback(
        self, 
        run_id: str, 
        key: str, 
        score: float, 
        comment: Optional[str] = None
    ):
        """Log feedback"""
        if not self.is_enabled or not self.client:
            return
        
        try:
            self.client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=comment
            )
            logger.info(f"LangSmith feedback logged: {key}={score}")
        except Exception as e:
            logger.error(f"Failed to log LangSmith feedback: {str(e)}")


# Global LangSmith manager instance
langsmith_manager = LangSmithManager()


def with_langsmith_tracing(
    name: Optional[str] = None,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None
):
    """
    Decorator: Add LangSmith tracing to function
    
    Args:
        name: Run name
        tags: Tag list
        metadata: Metadata
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not langsmith_manager.is_enabled:
                return func(*args, **kwargs)
            
            run_name = name or f"{func.__module__}.{func.__name__}"
            
            # Create run record
            run = langsmith_manager.create_run(
                name=run_name,
                inputs={"args": str(args), "kwargs": str(kwargs)},
                run_type="chain",
                tags=tags or [],
                extra=metadata or {}
            )
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Update run record
                if run:
                    langsmith_manager.update_run(
                        run_id=run.id,
                        outputs={"result": str(result)},
                        end_time=None  # Automatically set end time
                    )
                
                return result
                
            except Exception as e:
                # Log error
                if run:
                    langsmith_manager.update_run(
                        run_id=run.id,
                        outputs={"error": str(e)},
                        error=str(e)
                    )
                raise
        
        return wrapper
    return decorator


def get_langsmith_config() -> Dict[str, Any]:
    """Get LangSmith configuration information"""
    return {
        "enabled": langsmith_manager.is_enabled,
        "project": settings.langchain_project,
        "endpoint": settings.langchain_endpoint,
        "tracing_v2": settings.langchain_tracing_v2
    } 