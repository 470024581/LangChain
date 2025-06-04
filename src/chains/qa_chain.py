from typing import Dict, Any, List, Optional
import logging

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from ..models.llm_factory import LLMFactory
from ..vectorstores.vector_store import VectorStoreManager
from ..prompts.prompt_templates import PromptTemplateManager, PromptFormatter
from ..memory.conversation_memory import ConversationMemoryManager

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
        self.llm = LLMFactory.create_openrouter_llm(model_name)
        self.use_memory = use_memory
        self.retriever_k = retriever_k
        
        # 初始化组件
        self.prompt_manager = PromptTemplateManager()
        self.memory_manager = ConversationMemoryManager() if use_memory else None
        
        # 获取检索器
        self.retriever = self._get_retriever()
        
        # 构建链
        self.chain = self._build_chain()
        
        logger.info(f"文档问答链初始化完成，使用模型: {model_name or 'default'}")
    
    def _get_retriever(self) -> BaseRetriever:
        """获取检索器"""
        if not self.vector_store_manager.vector_store:
            self.vector_store_manager.get_or_create_vector_store()
        
        return self.vector_store_manager.get_retriever(k=self.retriever_k)
    
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
            
            # 准备输入
            if self.use_memory:
                input_data = {
                    "question": question,
                    "session_id": session_id
                }
            else:
                input_data = question
            
            # 调用链
            answer = self.chain.invoke(input_data)
            
            # 获取相关文档
            relevant_docs = self.retriever.get_relevant_documents(question)
            
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
        """获取相关文档"""
        return self.retriever.get_relevant_documents(question)
    
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
        self.llm = LLMFactory.create_openrouter_llm(model_name)
        self.retriever_k = retriever_k
        
        # 初始化组件
        self.prompt_manager = PromptTemplateManager()
        self.memory_manager = ConversationMemoryManager()
        
        # 获取检索器
        self.retriever = self._get_retriever()
        
        # 构建链
        self.standalone_question_chain = self._build_standalone_question_chain()
        self.qa_chain = self._build_qa_chain()
        
        logger.info("对话式检索问答链初始化完成")
    
    def _get_retriever(self) -> BaseRetriever:
        """获取检索器"""
        if not self.vector_store_manager.vector_store:
            self.vector_store_manager.get_or_create_vector_store()
        
        return self.vector_store_manager.get_retriever(k=self.retriever_k)
    
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
            relevant_docs = self.retriever.get_relevant_documents(standalone_question)
            
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