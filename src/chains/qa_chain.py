from typing import Dict, Any, List, Optional
import logging

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.config import RunnableConfig
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from ..models.llm_factory import LLMFactory
from ..vectorstores.vector_store import VectorStoreManager
from ..prompts.prompt_templates import PromptTemplateManager, PromptFormatter
from ..memory.conversation_memory import ConversationMemoryManager
from ..utils.langsmith_utils import langsmith_manager, with_langsmith_tracing
from ..config.settings import settings

logger = logging.getLogger(__name__)


class DocumentQAChain:
    """文档问答链 - 使用LCEL构建"""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = None,
        use_memory: bool = True,
        retriever_k: int = 4
    ):
        """
        初始化问答链
        
        Args:
            vector_store_manager: 向量存储管理器
            model_name: 使用的模型名称
            use_memory: 是否使用记忆功能
            retriever_k: 检索器返回的文档数量
        """
        self.vector_store_manager = vector_store_manager
        self.use_memory = use_memory
        self.retriever_k = retriever_k
        
        # 初始化LLM和提示词管理器
        self.llm = LLMFactory.create_llm(model_name)
        self.prompt_manager = PromptTemplateManager()
        self.memory_manager = ConversationMemoryManager() if use_memory else None
        
        # 获取检索器
        self.retriever = self._get_retriever()
        
        # 构建链
        self.chain = self._build_chain()
        
        # 配置 LangSmith 追踪
        self._configure_langsmith()
        
        logger.info(f"文档问答链初始化完成，使用模型: {model_name or 'default'}")
        if langsmith_manager.is_enabled:
            logger.info("LangSmith 追踪已启用")
    
    def _get_retriever(self) -> BaseRetriever:
        """获取带有Rerank功能的检索器"""
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
    
    def _build_chain(self) -> Runnable:
        """构建LCEL链"""
        # 文档格式化函数
        def format_docs(docs: List[Document]) -> str:
            return PromptFormatter.format_documents(docs)
        
        if self.use_memory:
            # 带记忆的链
            prompt = self.prompt_manager.get_chat_qa_prompt()
            
            def get_chat_history(inputs: Dict[str, Any]) -> List:
                session_id = inputs.get("session_id", "default")
                return self.memory_manager.get_chat_history(session_id)
            
            def get_question(inputs: Dict[str, Any]) -> str:
                return inputs.get("question", "")
            
            # 构建带记忆的并行链
            chain = (
                RunnableParallel({
                    "context": RunnableLambda(get_question) | self.retriever | format_docs,
                    "question": RunnableLambda(get_question),
                    "chat_history": RunnableLambda(get_chat_history)
                })
                | prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            # 不带记忆的简单链
            prompt = self.prompt_manager.get_qa_prompt()
            
            chain = (
                RunnableParallel({
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough()
                })
                | prompt
                | self.llm
                | StrOutputParser()
            )
        
        return chain
    
    def _configure_langsmith(self):
        """配置 LangSmith 追踪"""
        if langsmith_manager.is_enabled:
            # 为链添加 LangSmith 回调
            callbacks = langsmith_manager.get_callbacks()
            if callbacks:
                self.langsmith_config = RunnableConfig(
                    callbacks=callbacks,
                    tags=["DocumentQAChain", f"memory_{self.use_memory}"],
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

    @with_langsmith_tracing(name="DocumentQAChain.invoke", tags=["qa", "invoke"])
    def invoke(
        self, 
        question: str, 
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用问答链
        
        Args:
            question: 用户问题
            session_id: 会话ID（如果使用记忆）
            **kwargs: 其他参数
        
        Returns:
            包含答案和相关信息的字典
        """
        try:
            logger.info(f"处理问题: {question[:100]}...")
            
            # 1. 获取相关文档
            relevant_docs = self.retriever.invoke(question)
            
            # 2. 构建上下文
            context = self._format_docs(relevant_docs)
            
            # 准备输入
            if self.use_memory:
                input_data = {
                    "question": question,
                    "session_id": session_id
                }
            else:
                input_data = question
            
            # 调用链（如果启用 LangSmith，使用配置）
            if self.langsmith_config:
                answer = self.chain.invoke(input_data, config=self.langsmith_config)
            else:
                answer = self.chain.invoke(input_data)
            
            # 保存到记忆（如果启用）
            if self.use_memory and self.memory_manager:
                self.memory_manager.add_message_pair(
                    user_message=question,
                    ai_message=answer,
                    session_id=session_id
                )
            
            result = {
                "answer": answer,
                "question": question,
                "relevant_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ],
                "session_id": session_id if self.use_memory else None
            }
            
            logger.info("问答处理完成")
            return result
            
        except Exception as e:
            logger.error(f"问答处理失败: {str(e)}")
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
        流式调用问答链
        
        Args:
            question: 用户问题
            session_id: 会话ID
            **kwargs: 其他参数
        
        Yields:
            答案的流式片段
        """
        try:
            logger.info(f"流式处理问题: {question[:100]}...")
            
            # 准备输入
            if self.use_memory:
                input_data = {
                    "question": question,
                    "session_id": session_id
                }
            else:
                input_data = question
            
            # 流式调用链
            full_answer = ""
            for chunk in self.chain.stream(input_data):
                full_answer += chunk
                yield chunk
            
            # 保存到记忆（如果启用）
            if self.use_memory and self.memory_manager:
                self.memory_manager.add_message_pair(
                    user_message=question,
                    ai_message=full_answer,
                    session_id=session_id
                )
            
        except Exception as e:
            logger.error(f"流式问答处理失败: {str(e)}")
            yield f"抱歉，处理您的问题时出现错误: {str(e)}"
    
    def get_relevant_documents(self, question: str) -> List[Document]:
        """获取与问题相关的文档"""
        logger.info(f"正在为问题检索相关文档: {question[:100]}...")
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
        return {}
    
    def update_retriever_k(self, k: int) -> None:
        """更新检索器返回的文档数量"""
        self.retriever_k = k
        self.retriever = self._get_retriever()
        self.chain = self._build_chain()
        logger.info(f"检索器文档数量更新为: {k}")

    def _format_docs(self, docs: List[Document]) -> str:
        # Implementation of _format_docs method
        pass

    def is_langsmith_enabled(self):
        return self.langsmith_config is not None


class ConversationalRetrievalChain:
    """对话式检索问答链"""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = None,
        retriever_k: int = 4
    ):
        """
        初始化对话式检索链
        
        Args:
            vector_store_manager: 向量存储管理器
            model_name: 使用的模型名称
            retriever_k: 检索器返回的文档数量
        """
        self.vector_store_manager = vector_store_manager
        self.model_name = model_name
        self.retriever_k = retriever_k
        
        # 初始化LLM和提示词管理器
        self.llm = LLMFactory.create_llm(model_name)
        self.prompt_manager = PromptTemplateManager()
        self.memory_manager = ConversationMemoryManager()
        
        # 获取检索器
        self.retriever = self._get_retriever()
        
        # 构建链
        self.standalone_question_chain = self._build_standalone_question_chain()
        self.qa_chain = self._build_qa_chain()
        
        logger.info("对话式检索问答链初始化完成")
    
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
    
    def _build_standalone_question_chain(self) -> Runnable:
        """构建独立问题生成链"""
        prompt = self.prompt_manager.get_standalone_question_prompt()
        
        def format_chat_history(inputs: Dict[str, Any]) -> str:
            session_id = inputs.get("session_id", "default")
            return self.memory_manager.format_chat_history_for_prompt(session_id)
        
        return (
            RunnableParallel({
                "chat_history": RunnableLambda(format_chat_history),
                "question": RunnablePassthrough()
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _build_qa_chain(self) -> Runnable:
        """构建问答链"""
        prompt = self.prompt_manager.get_qa_prompt()
        
        def format_docs(docs: List[Document]) -> str:
            return PromptFormatter.format_documents(docs)
        
        return (
            RunnableParallel({
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def invoke(
        self, 
        question: str, 
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用对话式检索问答链
        
        Args:
            question: 用户问题
            session_id: 会话ID
            **kwargs: 其他参数
        
        Returns:
            包含答案和相关信息的字典
        """
        try:
            logger.info(f"对话式处理问题: {question[:100]}...")
            
            # 1. 生成独立问题
            standalone_question = self.standalone_question_chain.invoke({
                "question": question,
                "session_id": session_id
            })
            
            logger.debug(f"独立问题: {standalone_question}")
            
            # 2. 使用独立问题进行问答
            answer = self.qa_chain.invoke(standalone_question)
            
            # 3. 获取相关文档
            relevant_docs = self.retriever.invoke(standalone_question)
            
            # 4. 保存到记忆
            self.memory_manager.add_message_pair(
                user_message=question,
                ai_message=answer,
                session_id=session_id
            )
            
            result = {
                "answer": answer,
                "question": question,
                "standalone_question": standalone_question,
                "relevant_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ],
                "session_id": session_id
            }
            
            logger.info("对话式问答处理完成")
            return result
            
        except Exception as e:
            logger.error(f"对话式问答处理失败: {str(e)}")
            return {
                "answer": f"抱歉，处理您的问题时出现错误: {str(e)}",
                "question": question,
                "relevant_documents": [],
                "session_id": session_id,
                "error": str(e)
            }

    def _get_standalone_question(self, question: str, chat_history: list) -> str:
        """根据对话历史，生成一个独立的、无须上下文就能理解的问题。"""
        if not chat_history:
            return question

        chain = self._get_standalone_question_chain()
        result = chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })
        return result

    def _get_standalone_question_chain(self):
        # 1. 获取相关文档
        relevant_docs = self.retriever.invoke(question)

        # 2. 构建上下文
        context = self._format_docs(relevant_docs)

        # 3. 生成独立问题
        standalone_question = self._get_standalone_question(question, chat_history)

        # 4. 获取相关文档
        relevant_docs = self.retriever.invoke(standalone_question)

        # 5. 构建上下文
        context = self._format_docs(relevant_docs)

        return standalone_question

    def get_relevant_documents(self, question: str) -> List[Document]:
        """获取与问题相关的文档"""
        logger.info(f"正在为问题检索相关文档: {question[:100]}...")
        return self.retriever.invoke(question)

    def _format_docs(self, docs: List[Document]) -> str:
        # Implementation of _format_docs method
        pass