"""
LangSmith 集成工具模块
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
    """LangSmith 管理器"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.tracer: Optional[LangChainTracer] = None
        self._is_enabled = False
        self._initialize()
    
    def _initialize(self):
        """初始化 LangSmith 连接"""
        try:
            # 检查是否有LangSmith API密钥
            if not settings.langsmith_api_key:
                logger.info("LangSmith API密钥未设置，跳过LangSmith初始化")
                self._is_enabled = False
                return
                
            if settings.langchain_tracing_v2 and settings.langsmith_api_key:
                # 强制设置环境变量（临时调试）
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
                os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
                os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
                
                logger.info(f"强制设置环境变量: PROJECT={settings.langchain_project}")
                
                # 初始化客户端
                self.client = Client(
                    api_url=settings.langchain_endpoint,
                    api_key=settings.langsmith_api_key
                )
                
                # 初始化追踪器
                self.tracer = LangChainTracer(
                    project_name=settings.langchain_project,
                    client=self.client
                )
                
                self._is_enabled = True
                logger.info(f"LangSmith 已启用，项目: {settings.langchain_project}")
                
                # 测试连接
                self._test_connection()
                
            else:
                logger.info("LangSmith 未启用")
                self._is_enabled = False
                
        except Exception as e:
            logger.warning(f"LangSmith 初始化失败，将禁用追踪: {str(e)}")
            self._is_enabled = False
    
    def _test_connection(self):
        """测试 LangSmith 连接"""
        try:
            if self.client:
                # 尝试获取项目信息
                projects = list(self.client.list_projects(limit=1))
                logger.info("LangSmith 连接测试成功")
        except Exception as e:
            logger.warning(f"LangSmith 连接测试失败: {str(e)}")
    
    @property
    def is_enabled(self) -> bool:
        """检查 LangSmith 是否启用"""
        return self._is_enabled
    
    def get_callback_manager(self) -> Optional[CallbackManager]:
        """获取回调管理器"""
        if self.is_enabled and self.tracer:
            return CallbackManager([self.tracer])
        return None
    
    def get_callbacks(self) -> list:
        """获取回调列表"""
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
        """创建运行记录"""
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
            logger.error(f"创建 LangSmith 运行记录失败: {str(e)}")
            return None
    
    def update_run(self, run_id: str, outputs: Dict[str, Any], **kwargs):
        """更新运行记录"""
        if not self.is_enabled or not self.client:
            return
        
        try:
            self.client.update_run(
                run_id=run_id,
                outputs=outputs,
                **kwargs
            )
        except Exception as e:
            logger.error(f"更新 LangSmith 运行记录失败: {str(e)}")
    
    def log_feedback(
        self, 
        run_id: str, 
        key: str, 
        score: float, 
        comment: Optional[str] = None
    ):
        """记录反馈"""
        if not self.is_enabled or not self.client:
            return
        
        try:
            self.client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=comment
            )
            logger.info(f"LangSmith 反馈已记录: {key}={score}")
        except Exception as e:
            logger.error(f"记录 LangSmith 反馈失败: {str(e)}")


# 全局 LangSmith 管理器实例
langsmith_manager = LangSmithManager()


def with_langsmith_tracing(
    name: Optional[str] = None,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None
):
    """
    装饰器：为函数添加 LangSmith 追踪
    
    Args:
        name: 运行名称
        tags: 标签列表
        metadata: 元数据
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not langsmith_manager.is_enabled:
                return func(*args, **kwargs)
            
            run_name = name or f"{func.__module__}.{func.__name__}"
            
            # 创建运行记录
            run = langsmith_manager.create_run(
                name=run_name,
                inputs={"args": str(args), "kwargs": str(kwargs)},
                run_type="chain",
                tags=tags or [],
                extra=metadata or {}
            )
            
            try:
                # 执行函数
                result = func(*args, **kwargs)
                
                # 更新运行记录
                if run:
                    langsmith_manager.update_run(
                        run_id=run.id,
                        outputs={"result": str(result)},
                        end_time=None  # 自动设置结束时间
                    )
                
                return result
                
            except Exception as e:
                # 记录错误
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
    """获取 LangSmith 配置信息"""
    return {
        "enabled": langsmith_manager.is_enabled,
        "project": settings.langchain_project,
        "endpoint": settings.langchain_endpoint,
        "tracing_v2": settings.langchain_tracing_v2
    } 