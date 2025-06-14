"""
SQL AI Agent 模块
基于 create_sql_agent 构建的智能SQL查询代理
"""

import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from langchain.agents import AgentType

from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentExecutor, initialize_agent
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts import PromptTemplate

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
        """创建 SQL Agent（使用自定义提示词）"""
        try:
            # 创建自定义系统提示词，解决输出解析问题
            system_message = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

IMPORTANT OUTPUT FORMAT RULES:
- Only one step at a time.
- Either output "Action: <action>" or "Final Answer: <answer>".
- Do not output both in the same response.
- When you have the final answer, ONLY output "Final Answer: <your answer>".
- When you need to use a tool, ONLY output "Action: <tool_name>" followed by "Action Input: <input>".

If the question does not seem related to the database, just return "I don't know" as the answer."""

            # 创建自定义提示词模板
            prompt = PromptTemplate.from_template(f"""{system_message}

Question: {{input}}
{{agent_scratchpad}}""")

            # 创建SQL数据库工具包
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            tools = toolkit.get_tools()

            # 使用initialize_agent创建代理
            agent_executor = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                prompt=prompt,
                verbose=self.verbose,
                handle_parsing_errors=True
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
        执行SQL查询（同步方法，兼容现有代码）
        
        Args:
            question: 用户的自然语言问题
            session_id: 会话ID（如果使用记忆）
            **kwargs: 其他参数
        
        Returns:
            包含查询结果的字典
        """
        logger.info(f"处理SQL查询: {question[:100]}...")

        if not self.llm:
            logger.error("LLM未初始化，SQL Agent查询无法执行")
            return {
                "query": question, 
                "query_type": "sql_agent", 
                "success": False,
                "answer": "SQL Agent功能无法执行，因为LLM未初始化。",
                "data": {"session_id": session_id},
                "error": "LLM未初始化，SQL Agent无法使用。"
            }

        try:
            # 检查是否为简单的数据库结构查询，这类查询不需要历史上下文
            simple_queries = [
                "表", "table", "数据库", "database", "结构", "schema", 
                "列", "column", "字段", "field", "有哪些", "什么表"
            ]
            is_simple_query = any(keyword in question.lower() for keyword in simple_queries)
            
            # 如果使用记忆且不是简单查询，从记忆中获取上下文
            chat_history = ""
            if self.use_memory and self.memory_manager and not is_simple_query:
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
                        logger.info("使用聊天历史上下文")
                else:
                    logger.info("没有找到聊天历史")
            elif is_simple_query:
                logger.info("检测到简单数据库查询，跳过历史上下文")
            
            logger.info(f"执行SQL Agent查询: {question}")
            # 参考示例代码：Agent的invoke方法期望一个带有"input"键的字典
            if self.langsmith_config:
                result = self.agent_executor.invoke(
                    {"input": question}, 
                    config=self.langsmith_config
                )
            else:
                result = self.agent_executor.invoke({"input": question})
            
            # 提取答案
            answer = result.get("output", "无法从SQL Agent获取答案。")
            logger.info(f"SQL Agent执行完成。答案: {answer}")
            
            # 保存到记忆
            if self.use_memory and self.memory_manager:
                self.memory_manager.add_message_pair(
                    user_message=question.split("当前问题：")[-1] if "当前问题：" in question else question,
                    ai_message=answer,
                    session_id=session_id
                )
            
            # 构建返回结果（参考示例代码的返回格式）
            response = {
                "query": question,
                "query_type": "sql_agent",
                "success": True,
                "answer": answer,
                "data": {
                    "session_id": session_id,
                    "intermediate_steps": result.get("intermediate_steps", [])
                }
            }
            
            logger.info("SQL查询处理完成")
            return response
            
        except Exception as e:
            error_msg = f"SQL查询失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "query": question, 
                "query_type": "sql_agent", 
                "success": False,
                "answer": f"执行SQL查询时发生错误: {str(e)}",
                "data": {"session_id": session_id},
                "error": str(e)
            }
    
    def query_sync(
        self, 
        question: str, 
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        同步执行SQL查询（向后兼容）
        
        Args:
            question: 用户的自然语言问题
            session_id: 会话ID（如果使用记忆）
            **kwargs: 其他参数
        
        Returns:
            包含查询结果的字典
        """
        logger.info(f"处理同步SQL查询: {question[:100]}...")

        if not self.llm:
            logger.error("LLM未初始化，SQL Agent查询无法执行")
            return {
                "query": question, 
                "query_type": "sql_agent", 
                "success": False,
                "answer": "SQL Agent功能无法执行，因为LLM未初始化。",
                "data": {"session_id": session_id},
                "error": "LLM未初始化。"
            }

        try:
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
            
            # 同步执行查询
            if self.langsmith_config:
                result = self.agent_executor.invoke(
                    {"input": question}, 
                    config=self.langsmith_config
                )
            else:
                result = self.agent_executor.invoke({"input": question})
            
            # 提取答案
            answer = result.get("output", "无法从SQL Agent获取答案。")
            logger.info(f"同步SQL Agent执行完成。答案: {answer}")
            
            # 保存到记忆
            if self.use_memory and self.memory_manager:
                self.memory_manager.add_message_pair(
                    user_message=question.split("当前问题：")[-1] if "当前问题：" in question else question,
                    ai_message=answer,
                    session_id=session_id
                )
            
            # 构建返回结果
            response = {
                "query": question,
                "query_type": "sql_agent",
                "success": True,
                "answer": answer,
                "data": {
                    "session_id": session_id,
                    "intermediate_steps": result.get("intermediate_steps", [])
                }
            }
            
            logger.info("同步SQL查询处理完成")
            return response
            
        except Exception as e:
            error_msg = f"同步SQL查询失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "query": question, 
                "query_type": "sql_agent", 
                "success": False,
                "answer": f"执行同步SQL查询时发生错误: {str(e)}",
                "data": {"session_id": session_id},
                "error": str(e)
            }
    
    async def query_table(
        self, 
        query: str, 
        table_name: str,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        查询指定表（参考示例代码实现方式）
        
        Args:
            query: 用户查询
            table_name: 指定的表名
            session_id: 会话ID
        
        Returns:
            查询结果字典
        """
        logger.info(f"查询指定表 {table_name}: {query}")

        if not self.llm:
            logger.error("LLM未初始化，SQL Agent查询无法执行")
            return {
                "query": query, 
                "query_type": "sql_agent", 
                "success": False,
                "answer": "SQL Agent功能无法执行，因为LLM未初始化。",
                "data": {"queried_table": table_name, "session_id": session_id},
                "error": "LLM未初始化。"
            }

        try:
            logger.info(f"为表 {table_name} 初始化SQLDatabase，使用数据库: {self.db_path}")
            # 参考示例代码：SQLDatabase连接到主数据库，但只包含指定的表
            db_uri = f"sqlite:///{self.db_path}"
            db = SQLDatabase.from_uri(db_uri, include_tables=[table_name])
            
            logger.info(f"为表 {table_name} 创建SQL Agent")
            # 参考示例代码：创建专门的SQL Agent
            sql_agent_executor = create_sql_agent(
                llm=self.llm, 
                db=db, 
                verbose=True, 
                handle_parsing_errors=True
            )
            
            logger.info(f"使用查询执行SQL Agent: {query}")
            # 参考示例代码：Agent的ainvoke方法期望一个带有"input"键的字典
            response = await sql_agent_executor.ainvoke({"input": query})
            
            answer = response.get("output", "无法从SQL Agent获取答案。")
            logger.info(f"表 {table_name} 的SQL Agent执行完成。答案: {answer}")
            
            return {
                "query": query, 
                "query_type": "sql_agent", 
                "success": True,
                "answer": answer,
                "data": {
                    "queried_table": table_name,
                    "session_id": session_id,
                    "intermediate_steps": response.get("intermediate_steps", [])
                }
            }

        except Exception as e:
            logger.error(f"表 {table_name} 的SQL Agent查询失败: {e}", exc_info=True)
            return {
                "query": query, 
                "query_type": "sql_agent", 
                "success": False,
                "answer": f"在表 '{table_name}' 中执行SQL查询时发生错误。",
                "data": {"queried_table": table_name, "session_id": session_id},
                "error": str(e)
            }

    async def get_database_info(self) -> Dict[str, Any]:
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


# 按照示例代码实现方式的便捷函数
async def get_sql_agent_response(
    query: str,
    db_path: str = None,
    model_name: str = None,
    table_name: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    按照示例代码实现方式获取SQL Agent响应的便捷函数
    
    Args:
        query: 用户查询
        db_path: 数据库路径
        model_name: 模型名称  
        table_name: 特定表名（用于单表查询）
        **kwargs: 其他参数
    
    Returns:
        查询结果字典
    """
    try:
        # 创建SQL Agent实例
        agent = SQLAgent(
            db_path=db_path,
            model_name=model_name,
            use_memory=False,  # 便捷函数不使用记忆
            verbose=False
        )
        
        # 如果指定了表名，使用查询特定表的方法
        if table_name:
            return await agent.query_table(query, table_name, session_id="convenience_function")
        else:
            # 否则使用常规查询（同步）
            return agent.query(query, session_id="convenience_function")
            
    except Exception as e:
        logger.error(f"便捷函数SQL Agent查询失败: {str(e)}")
        return {
            "query": query,
            "query_type": "sql_agent",
            "success": False,
            "answer": f"SQL查询执行失败: {str(e)}",
            "data": {"queried_table": table_name} if table_name else {},
            "error": str(e)
        } 