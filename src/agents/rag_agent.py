"""
基于Agent的RAG (Retrieval-Augmented Generation) 实现

这个模块实现了使用LangChain内置Agent的RAG系统，
与现有的Chain版本保持相同的逻辑和接口。
"""

import logging
from typing import Dict, Any, List, Optional, Union
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from src.vectorstores.vector_store import VectorStoreManager
from src.models.llm_factory import LLMFactory
from src.prompts.prompt_templates import PromptTemplateManager, PromptFormatter
from src.memory.conversation_memory import ConversationMemoryManager
from src.utils.langsmith_utils import langsmith_manager, with_langsmith_tracing
from src.config.settings import settings

logger = logging.getLogger(__name__)


class DocumentQAAgent:
    """
    基于Agent的文档问答系统
    
    这个类实现了与DocumentQAChain相同的功能，但使用LangChain的Agent框架。
    """
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = None,
        use_memory: bool = True,
        retriever_k: int = 4
    ):
        """
        初始化Agent
        
        Args:
            vector_store_manager: 向量存储管理器
            model_name: 模型名称
            use_memory: 是否使用记忆
            retriever_k: 检索器返回的文档数量
        """
        self.vector_store_manager = vector_store_manager
        self.model_name = model_name
        self.use_memory = use_memory
        self.retriever_k = retriever_k
        
        # 初始化组件
        self.llm = LLMFactory.create_llm(model_name)
        self.prompt_manager = PromptTemplateManager()
        self.memory_manager = ConversationMemoryManager() if use_memory else None
        
        # 初始化检索器
        self.retriever = self._get_retriever()
        
        # 构建Agent
        self.agent_executor = self._build_agent()
        
        # 配置LangSmith
        self._configure_langsmith()
        
        logger.info("DocumentQAAgent 初始化完成")
    
    def _get_retriever(self) -> BaseRetriever:
        """获取带有Rerank功能的检索器"""
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
        
        if not self.vector_store_manager.vector_store:
            self.vector_store_manager.get_or_create_vector_store()
        
        base_retriever = self.vector_store_manager.get_retriever(k=self.retriever_k)
        
        # 初始化FlashrankRerank
        reranker = FlashrankRerank(model=settings.reranker_model, top_n=3)
        
        # 创建带有上下文压缩的检索器
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=base_retriever
        )
        
        logger.info("已创建带有CrossEncoder Rerank功能的压缩检索器")
        return compression_retriever
    
    def _create_tools(self) -> List[Tool]:
        """创建Agent工具"""
        tools = [
            Tool(
                name="document_retrieval",
                description="搜索并检索与问题相关的文档内容。用于回答需要基于文档知识的问题。",
                func=self.retriever.invoke
            )
        ]
        return tools
    
    def _build_agent(self) -> AgentExecutor:
        """构建Agent执行器"""
        tools = self._create_tools()
        
        # 使用简单的系统消息而不是复杂的提示模板
        system_message = """你是一个智能文档问答助手。你需要基于检索到的文档内容来回答用户问题。

工作流程：
1. 使用 document_retrieval 工具一次来搜索相关文档。
2. 分析检索到的文档内容，并基于此生成最终答案。

**非常重要**: 如果你决定使用工具，你的回答**必须**只包含工具调用的JSON，不要有任何其他文字。一旦获得工具返回的信息，再组织语言回答问题。如果文档中没有相关信息，请明确说明。
"""

        # 创建提示模板，让LangChain自动处理工具部分
        if self.use_memory:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
        
        # 创建tool-calling agent
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        # 创建Agent执行器
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True,
            return_intermediate_steps=True, # 确保返回中间步骤
        )
        
        return agent_executor
    
    def _configure_langsmith(self):
        """配置 LangSmith 追踪"""
        if langsmith_manager.is_enabled:
            callbacks = langsmith_manager.get_callbacks()
            if callbacks:
                self.langsmith_config = RunnableConfig(
                    callbacks=callbacks,
                    tags=["DocumentQAAgent", f"memory_{self.use_memory}"],
                    metadata={
                        "retriever_k": self.retriever_k,
                        "use_memory": self.use_memory,
                        "model": getattr(self.llm, 'model_name', 'unknown')
                    }
                )
            else:
                self.langsmith_config = None
        else:
            self.langsmith_config = None
    
    def _prepare_input(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """准备Agent输入"""
        input_data = {"input": question}
        
        if self.use_memory and self.memory_manager:
            # 添加聊天历史
            chat_history = self.memory_manager.get_chat_history(session_id)
            input_data["chat_history"] = chat_history
        
        return input_data
    
    @with_langsmith_tracing(name="DocumentQAAgent.invoke", tags=["qa", "agent"])
    def invoke(
        self, 
        question: str, 
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用问答Agent
        
        Args:
            question: 用户问题
            session_id: 会话ID（如果使用记忆）
            **kwargs: 其他参数
        
        Returns:
            包含答案和相关信息的字典
        """
        try:
            logger.info(f"Agent处理问题: {question[:100]}...")
            
            # 准备输入
            input_data = self._prepare_input(question, session_id)
            
            # 调用Agent（如果启用 LangSmith，使用配置）
            if self.langsmith_config:
                result = self.agent_executor.invoke(input_data, config=self.langsmith_config)
            else:
                result = self.agent_executor.invoke(input_data)
            
            answer = result.get("output", "")
            
            # 从中间步骤中提取相关文档
            relevant_docs = []
            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    if action.tool == "document_retrieval" and isinstance(observation, list):
                        relevant_docs.extend(doc for doc in observation if isinstance(doc, Document))

            # 如果没有从中间步骤找到，作为后备方案再检索一次
            if not relevant_docs and question:
                logger.warning("无法从Agent中间步骤提取文档，执行后备检索。")
                relevant_docs = self.retriever.invoke(question)
            
            # 保存到记忆（如果启用）
            if self.use_memory and self.memory_manager:
                self.memory_manager.add_message_pair(
                    user_message=question,
                    ai_message=answer,
                    session_id=session_id
                )
            
            response = {
                "answer": answer,
                "question": question,
                "relevant_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ],
                "session_id": session_id if self.use_memory else None,
                "intermediate_steps": result.get("intermediate_steps", [])
            }
            
            logger.info("Agent问答处理完成")
            return response
            
        except Exception as e:
            logger.error(f"Agent问答处理失败: {str(e)}")
            return {
                "answer": f"抱歉，处理您的问题时出现错误: {str(e)}",
                "question": question,
                "relevant_documents": [],
                "session_id": session_id if self.use_memory else None,
                "error": str(e)
            }
    
    async def ainvoke(
        self, 
        question: str, 
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        异步调用问答Agent
        
        Args:
            question: 用户问题
            session_id: 会话ID
            **kwargs: 其他参数
        
        Returns:
            包含答案和相关信息的字典
        """
        try:
            logger.info(f"Agent异步处理问题: {question[:100]}...")
            
            # 准备输入
            input_data = self._prepare_input(question, session_id)
            
            # 异步调用Agent
            if self.langsmith_config:
                result = await self.agent_executor.ainvoke(input_data, config=self.langsmith_config)
            else:
                result = await self.agent_executor.ainvoke(input_data)
            
            answer = result.get("output", "")
            
            # 异步地从中间步骤提取文档
            relevant_docs = []
            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    if action.tool == "document_retrieval" and isinstance(observation, list):
                        relevant_docs.extend(doc for doc in observation if isinstance(doc, Document))
            
            if not relevant_docs and question:
                logger.warning("无法从Agent中间步骤提取文档，执行后备检索。")
                relevant_docs = await self.retriever.ainvoke(question)

            # 保存到记忆（如果启用）
            if self.use_memory and self.memory_manager:
                self.memory_manager.add_message_pair(
                    user_message=question,
                    ai_message=answer,
                    session_id=session_id
                )
            
            response = {
                "answer": answer,
                "question": question,
                "relevant_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ],
                "session_id": session_id if self.use_memory else None,
                "intermediate_steps": result.get("intermediate_steps", [])
            }
            
            logger.info("Agent异步问答处理完成")
            return response
            
        except Exception as e:
            logger.error(f"Agent异步问答处理失败: {str(e)}")
            return {
                "answer": f"抱歉，处理您的问题时出现错误: {str(e)}",
                "question": question,
                "relevant_documents": [],
                "session_id": session_id if self.use_memory else None,
                "error": str(e)
            }
    
    def stream(
        self, 
        question: str, 
        session_id: str = "default",
        **kwargs
    ):
        """
        流式调用问答Agent
        
        Args:
            question: 用户问题
            session_id: 会话ID
            **kwargs: 其他参数
        
        Yields:
            Agent执行过程中的流式输出
        """
        try:
            logger.info(f"Agent流式处理问题: {question[:100]}...")
            
            # 准备输入
            input_data = self._prepare_input(question, session_id)
            
            full_answer = ""
            # 流式调用Agent
            for chunk in self.agent_executor.stream(input_data):
                if "output" in chunk:
                    content = chunk["output"]
                    full_answer += content
                    yield content
                elif "intermediate_step" in chunk:
                    # 也可以输出中间步骤
                    step = chunk["intermediate_step"]
                    yield f"\n[Agent思考]: {step}\n"
            
            # 保存到记忆（如果启用）
            if self.use_memory and self.memory_manager:
                self.memory_manager.add_message_pair(
                    user_message=question,
                    ai_message=full_answer,
                    session_id=session_id
                )
            
        except Exception as e:
            logger.error(f"Agent流式问答处理失败: {str(e)}")
            yield f"抱歉，处理您的问题时出现错误: {str(e)}"
    
    def get_relevant_documents(self, question: str) -> List[Document]:
        """获取与问题相关的文档"""
        logger.info(f"Agent 正在为问题检索相关文档: {question[:100]}...")
        return self.retriever.invoke(question)
    
    def clear_memory(self, session_id: str = "default") -> None:
        """清空指定会话的记忆"""
        if self.use_memory and self.memory_manager:
            self.memory_manager.clear_memory(session_id)
            logger.info(f"已清空会话 {session_id} 的记忆")
    
    def get_memory_stats(self, session_id: str = "default") -> Dict[str, Any]:
        """获取记忆统计信息"""
        if self.use_memory and self.memory_manager:
            return self.memory_manager.get_memory_stats(session_id)
        else:
            return {"use_memory": False}
    
    def update_retriever_k(self, k: int) -> None:
        """更新检索器返回的文档数量"""
        self.retriever_k = k
        self.retriever = self._get_retriever()
        # 重新构建Agent
        self.agent_executor = self._build_agent()
        logger.info(f"检索器文档数量已更新为: {k}")


class ConversationalRetrievalAgent:
    """
    基于Agent的对话式检索链
    
    实现与ConversationalRetrievalChain相同的功能，使用Agent架构。
    """
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = None,
        retriever_k: int = 4
    ):
        """
        初始化对话式检索Agent
        
        Args:
            vector_store_manager: 向量存储管理器
            model_name: 模型名称
            retriever_k: 检索器返回的文档数量
        """
        self.vector_store_manager = vector_store_manager
        self.model_name = model_name
        self.retriever_k = retriever_k
        
        # 初始化组件
        self.llm = LLMFactory.create_llm(model_name)
        self.prompt_manager = PromptTemplateManager()
        
        # 初始化检索器
        self.retriever = self._get_retriever()
        
        # 构建Agent
        self.agent_executor = self._build_agent()
        
        logger.info("ConversationalRetrievalAgent 初始化完成")
    
    def _get_retriever(self) -> BaseRetriever:
        """获取带有Rerank功能的检索器"""
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
        
        if not self.vector_store_manager.vector_store:
            self.vector_store_manager.get_or_create_vector_store()
            
        base_retriever = self.vector_store_manager.get_retriever(k=self.retriever_k)
        
        # 初始化FlashrankRerank
        reranker = FlashrankRerank(model=settings.reranker_model, top_n=3)
        
        # 创建带有上下文压缩的检索器
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=base_retriever
        )
        
        logger.info("已创建带有CrossEncoder Rerank功能的压缩检索器")
        return compression_retriever
    
    def _create_tools(self) -> List[Tool]:
        """创建Agent工具"""
        tools = [
            Tool(
                name="document_retrieval",
                description="搜索并检索与问题相关的文档内容。",
                func=self.retriever.invoke,
            )
        ]
        return tools
    
    def _generate_standalone_question(self, input_with_history: str) -> str:
        """生成独立问题"""
        # 这里可以实现更复杂的逻辑来从对话历史中提取独立问题
        # 简单实现：直接返回最后的问题
        lines = input_with_history.strip().split('\n')
        return lines[-1] if lines else input_with_history
    
    def _build_agent(self) -> AgentExecutor:
        """构建Agent执行器"""
        tools = self._create_tools()
        
        system_message = """你是一个对话式文档检索助手。你需要：

1. 分析用户的问题和对话历史，如果需要，重新表述问题以便更好地检索。
2. 使用 document_retrieval 工具检索相关文档。
3. 基于检索结果和对话上下文回答问题。

**非常重要**: 如果你决定使用工具，你的回答**必须**只包含工具调用的JSON，不要有任何其他文字。一旦获得工具返回的信息，再组织语言回答问题。如果文档中没有相关信息，请明确说明。"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        
        return agent_executor
    
    def invoke(
        self,
        question: str,
        chat_history: List[tuple] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用对话式检索Agent
        
        Args:
            question: 用户问题
            chat_history: 对话历史
            **kwargs: 其他参数
        
        Returns:
            包含答案和相关信息的字典
        """
        try:
            logger.info(f"对话式Agent处理问题: {question[:100]}...")
            
            # 准备输入，包含对话历史
            if chat_history:
                history_str = "\n".join([f"Human: {h}\nAssistant: {a}" for h, a in chat_history])
                input_text = f"对话历史:\n{history_str}\n\n当前问题: {question}"
            else:
                input_text = question
            
            input_data = {"input": input_text}
            
            # 调用Agent
            result = self.agent_executor.invoke(input_data)
            answer = result.get("output", "")
            
            # 从中间步骤中提取相关文档
            relevant_docs = []
            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    if action.tool == "document_retrieval" and isinstance(observation, list):
                        relevant_docs.extend(doc for doc in observation if isinstance(doc, Document))

            if not relevant_docs and question:
                logger.warning("无法从Agent中间步骤提取文档，执行后备检索。")
                relevant_docs = self.retriever.invoke(question)
            
            response = {
                "answer": answer,
                "question": question,
                "relevant_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ],
                "chat_history": chat_history,
                "intermediate_steps": result.get("intermediate_steps", [])
            }
            
            logger.info("对话式Agent问答处理完成")
            return response
            
        except Exception as e:
            logger.error(f"对话式Agent问答处理失败: {str(e)}")
            return {
                "answer": f"抱歉，处理您的问题时出现错误: {str(e)}",
                "question": question,
                "relevant_documents": [],
                "chat_history": chat_history,
                "error": str(e)
            }