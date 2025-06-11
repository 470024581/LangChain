"""
SQL AI Agent 模块
基于 create_sql_agent 构建的智能SQL查询代理
"""

import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentExecutor
from langchain_core.runnables.config import RunnableConfig

from ..models.llm_factory import LLMFactory
from ..memory.conversation_memory import ConversationMemoryManager
from ..utils.langsmith_utils import langsmith_manager, with_langsmith_tracing
from ..config.settings import settings

logger = logging.getLogger(__name__)


class SQLAgent:
    """SQL AI Agent - 基于 create_sql_agent 构建"""
    
    def __init__(
        self,
        db_path: str = None,
        model_name: str = None,
        use_memory: bool = True,
        verbose: bool = True
    ):
        """
        初始化 SQL Agent
        
        Args:
            db_path: SQLite 数据库文件路径
            model_name: 使用的模型名称
            use_memory: 是否使用记忆功能
            verbose: 是否显示详细输出
        """
        self.db_path = db_path or self._get_default_db_path()
        self.model_name = model_name
        self.use_memory = use_memory
        self.verbose = verbose
        
        # 验证数据库文件
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")
        
        # 初始化组件
        self.llm = LLMFactory.create_llm(model_name)
        self.db = self._create_database_connection()
        self.memory_manager = ConversationMemoryManager() if use_memory else None
        
        # 创建 SQL Agent
        self.agent_executor = self._create_sql_agent()
        
        # 配置 LangSmith
        self._configure_langsmith()
        
        logger.info(f"SQL Agent 初始化完成，数据库: {self.db_path}")
        if langsmith_manager.is_enabled:
            logger.info("LangSmith 追踪已启用")
    
    def _get_default_db_path(self) -> str:
        """获取默认数据库路径"""
        # 从项目根目录查找数据库文件
        base_path = Path(__file__).parent.parent.parent
        db_path = base_path / "data" / "database" / "erp.db"
        return str(db_path)
    
    def _create_database_connection(self) -> SQLDatabase:
        """创建数据库连接"""
        try:
            # 构建SQLite连接URI
            db_uri = f"sqlite:///{self.db_path}"
            
            # 创建SQLDatabase实例
            db = SQLDatabase.from_uri(
                db_uri,
                # 设置采样行数，用于提供表数据示例
                sample_rows_in_table_info=3,
                # 设置包含的表（如果需要限制）
                include_tables=['products', 'inventory', 'sales']
            )
            
            logger.info(f"数据库连接成功: {db_uri}")
            return db
            
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            raise
    
    def _create_sql_agent(self) -> AgentExecutor:
        """创建 SQL Agent"""
        try:
            # 使用 LangChain 的 create_sql_agent 函数
            agent_executor = create_sql_agent(
                llm=self.llm,
                db=self.db,
                verbose=self.verbose,
                handle_parsing_errors=True,
                # 设置 agent 类型
                agent_type="tool-calling",
                # 自定义前缀来改善中文支持
                prefix="""
你是一个专业的数据库查询助手。你的任务是理解用户的自然语言问题，并生成相应的SQL查询来获取所需信息。

你有以下几个工具可以使用：
1. sql_db_list_tables: 列出数据库中的所有表
2. sql_db_schema: 获取表的结构信息
3. sql_db_query: 执行SQL查询并返回结果
4. sql_db_query_checker: 检查SQL查询的正确性

数据库说明：
- products表：包含产品信息（product_id, product_name, category, unit_price）
- inventory表：包含库存信息（product_id, stock_level, last_updated）
- sales表：包含销售记录（sale_id, product_id, product_name, quantity_sold, price_per_unit, total_amount, sale_date）

请按照以下步骤处理用户问题：
1. 理解用户的问题和意图
2. 确定需要查询哪些表
3. 生成正确的SQL语句
4. 执行查询并获取结果
5. 用自然语言总结结果

注意事项：
- 始终检查SQL语句的正确性
- 如果查询返回空结果，请说明可能的原因
- 对于复杂查询，可以使用JOIN连接多个表
- 金额和数量请保留适当的小数位数
""",
                # 设置后缀
                suffix="开始吧！记住首先了解表结构，然后根据用户问题生成合适的SQL查询。",
                # 设置格式说明
                format_instructions="""使用以下格式回答：

问题：用户的原始问题
思考：你需要做什么来回答这个问题
行动：要采取的行动，应该是[sql_db_list_tables, sql_db_schema, sql_db_query, sql_db_query_checker]之一
行动输入：行动的输入
观察：行动的结果
... (这个思考/行动/行动输入/观察可以重复多次)
思考：我现在知道最终答案了
最终答案：对原始输入问题的最终答案"""
            )
            
            logger.info("SQL Agent 创建成功")
            return agent_executor
            
        except Exception as e:
            logger.error(f"SQL Agent 创建失败: {str(e)}")
            raise
    
    def _configure_langsmith(self):
        """配置 LangSmith 追踪"""
        if langsmith_manager.is_enabled:
            callbacks = langsmith_manager.get_callbacks()
            if callbacks:
                self.langsmith_config = RunnableConfig(
                    callbacks=callbacks,
                    tags=["SQLAgent", f"memory_{self.use_memory}"],
                    metadata={
                        "db_path": self.db_path,
                        "use_memory": self.use_memory,
                        "model": getattr(self.llm, 'model_name', 'unknown')
                    }
                )
            else:
                self.langsmith_config = None
        else:
            self.langsmith_config = None
    
    @with_langsmith_tracing(name="SQLAgent.query", tags=["sql", "query"])
    def query(
        self, 
        question: str, 
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行SQL查询
        
        Args:
            question: 用户的自然语言问题
            session_id: 会话ID（如果使用记忆）
            **kwargs: 其他参数
        
        Returns:
            包含查询结果的字典
        """
        try:
            logger.info(f"处理SQL查询: {question[:100]}...")
            
            # 如果使用记忆，从记忆中获取上下文
            chat_history = ""
            if self.use_memory and self.memory_manager:
                history = self.memory_manager.get_chat_history(session_id)
                if history:
                    # 格式化聊天历史
                    history_texts = []
                    for msg in history[-4:]:  # 只取最近4轮对话
                        if hasattr(msg, 'content'):
                            role = "用户" if msg.type == "human" else "助手"
                            history_texts.append(f"{role}: {msg.content}")
                    if history_texts:
                        chat_history = "\n".join(history_texts)
                        question = f"聊天历史：\n{chat_history}\n\n当前问题：{question}"
            
            # 执行查询
            if self.langsmith_config:
                result = self.agent_executor.invoke(
                    {"input": question}, 
                    config=self.langsmith_config
                )
            else:
                result = self.agent_executor.invoke({"input": question})
            
            # 提取答案
            answer = result.get("output", "抱歉，无法获取查询结果。")
            
            # 保存到记忆
            if self.use_memory and self.memory_manager:
                self.memory_manager.add_message_pair(
                    user_message=question.split("当前问题：")[-1] if "当前问题：" in question else question,
                    ai_message=answer,
                    session_id=session_id
                )
            
            # 构建返回结果
            response = {
                "answer": answer,
                "question": question,
                "session_id": session_id,
                "intermediate_steps": result.get("intermediate_steps", []),
                "success": True
            }
            
            logger.info("SQL查询处理完成")
            return response
            
        except Exception as e:
            error_msg = f"SQL查询失败: {str(e)}"
            logger.error(error_msg)
            
            return {
                "answer": error_msg,
                "question": question,
                "session_id": session_id,
                "success": False,
                "error": str(e)
            }
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        try:
            # 获取表列表
            tables = self.db.get_usable_table_names()
            
            # 获取表结构
            table_info = {}
            for table in tables:
                table_info[table] = self.db.get_table_info([table])
            
            return {
                "database_path": self.db_path,
                "tables": tables,
                "table_info": table_info,
                "dialect": self.db.dialect
            }
            
        except Exception as e:
            logger.error(f"获取数据库信息失败: {str(e)}")
            return {"error": str(e)}
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> Dict[str, Any]:
        """获取表的样例数据"""
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            result = self.db.run(query)
            
            return {
                "table": table_name,
                "sample_data": result,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"获取样例数据失败: {str(e)}")
            return {"error": str(e)}
    
    def clear_memory(self, session_id: str = "default") -> bool:
        """清除会话记忆"""
        if self.use_memory and self.memory_manager:
            return self.memory_manager.clear_memory(session_id)
        return True
    
    def get_memory_stats(self, session_id: str = "default") -> Dict[str, Any]:
        """获取记忆统计信息"""
        if self.use_memory and self.memory_manager:
            return self.memory_manager.get_memory_stats(session_id)
        return {}


# 便捷函数
def create_sql_agent_simple(
    db_path: str = None,
    model_name: str = None,
    **kwargs
) -> SQLAgent:
    """
    创建简单的SQL Agent实例
    
    Args:
        db_path: 数据库文件路径
        model_name: 模型名称
        **kwargs: 其他参数
    
    Returns:
        SQLAgent实例
    """
    return SQLAgent(
        db_path=db_path,
        model_name=model_name,
        **kwargs
    ) 