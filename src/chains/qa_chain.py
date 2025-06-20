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
    """Document Q&A Chain - Built with LCEL"""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = None,
        use_memory: bool = True,
        retriever_k: int = 4
    ):
        """
        Initialize Q&A chain
        
        Args:
            vector_store_manager: Vector store manager
            model_name: Model name to use
            use_memory: Whether to use memory function
            retriever_k: Number of documents returned by retriever
        """
        self.vector_store_manager = vector_store_manager
        self.use_memory = use_memory
        self.retriever_k = retriever_k
        
        # Initialize LLM and prompt manager
        self.llm = LLMFactory.create_llm(model_name)
        self.prompt_manager = PromptTemplateManager()
        self.memory_manager = ConversationMemoryManager() if use_memory else None
        
        # Get retriever
        self.retriever = self._get_retriever()
        
        # Build chain
        self.chain = self._build_chain()
        
        # Configure LangSmith tracing
        self._configure_langsmith()
        
        logger.info(f"Document Q&A chain initialization completed, using model: {model_name or 'default'}")
        if langsmith_manager.is_enabled:
            logger.info("LangSmith tracing enabled")
    
    def _get_retriever(self) -> BaseRetriever:
        """Get retriever with Rerank functionality"""
        if not self.vector_store_manager.vector_store:
            self.vector_store_manager.get_or_create_vector_store()
        
        base_retriever = self.vector_store_manager.get_retriever(k=self.retriever_k)
        
        # Initialize FlashrankRerank
        reranker = FlashrankRerank(model=settings.reranker_model, top_n=3)
        
        # Create retriever with contextual compression
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=base_retriever
        )
        
        logger.info("Created compression retriever with CrossEncoder Rerank functionality")
        return compression_retriever
    
    def _build_chain(self) -> Runnable:
        """Build LCEL chain"""
        # Document formatting function
        def format_docs(docs: List[Document]) -> str:
            return PromptFormatter.format_documents(docs)
        
        if self.use_memory:
            # Chain with memory
            prompt = self.prompt_manager.get_chat_qa_prompt()
            
            def get_chat_history(inputs: Dict[str, Any]) -> List:
                session_id = inputs.get("session_id", "default")
                return self.memory_manager.get_chat_history(session_id)
            
            def get_question(inputs: Dict[str, Any]) -> str:
                return inputs.get("question", "")
            
            # Build parallel chain with memory
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
            # Simple chain without memory
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
        """Configure LangSmith tracing"""
        if langsmith_manager.is_enabled:
            # Add LangSmith callbacks to chain
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
        Invoke Q&A chain
        
        Args:
            question: User question
            session_id: Session ID (if using memory)
            **kwargs: Other parameters
        
        Returns:
            Dictionary containing answer and related information
        """
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # 1. Get relevant documents
            relevant_docs = self.retriever.invoke(question)
            
            # 2. Build context
            context = self._format_docs(relevant_docs)
            
            # Prepare input
            if self.use_memory:
                input_data = {
                    "question": question,
                    "session_id": session_id
                }
            else:
                input_data = question
            
            # Invoke chain (use config if LangSmith is enabled)
            if self.langsmith_config:
                answer = self.chain.invoke(input_data, config=self.langsmith_config)
            else:
                answer = self.chain.invoke(input_data)
            
            # Save to memory (if enabled)
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
            
            logger.info("Q&A processing completed")
            return result
            
        except Exception as e:
            logger.error(f"Q&A processing failed: {str(e)}")
            return {
                "answer": f"Sorry, an error occurred while processing your question: {str(e)}",
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
        Stream invoke Q&A chain
        
        Args:
            question: User question
            session_id: Session ID
            **kwargs: Other parameters
        
        Yields:
            Streaming answer fragments
        """
        try:
            logger.info(f"Stream processing question: {question[:100]}...")
            
            # Prepare input
            if self.use_memory:
                input_data = {
                    "question": question,
                    "session_id": session_id
                }
            else:
                input_data = question
            
            # Stream invoke chain
            full_answer = ""
            for chunk in self.chain.stream(input_data):
                full_answer += chunk
                yield chunk
            
            # Save to memory (if enabled)
            if self.use_memory and self.memory_manager:
                self.memory_manager.add_message_pair(
                    user_message=question,
                    ai_message=full_answer,
                    session_id=session_id
                )
            
        except Exception as e:
            logger.error(f"Stream Q&A processing failed: {str(e)}")
            yield f"Sorry, an error occurred while processing your question: {str(e)}"
    
    def get_relevant_documents(self, question: str) -> List[Document]:
        """Get documents relevant to the question"""
        logger.info(f"Retrieving relevant documents for question: {question[:100]}...")
        return self.retriever.invoke(question)
    
    def clear_memory(self, session_id: str = "default") -> None:
        """Clear memory for specified session"""
        if self.use_memory and self.memory_manager:
            self.memory_manager.clear_memory(session_id)
            logger.info(f"Cleared memory for session {session_id}")
    
    def get_memory_stats(self, session_id: str = "default") -> Dict[str, Any]:
        """Get memory statistics"""
        if self.use_memory and self.memory_manager:
            return self.memory_manager.get_memory_stats(session_id)
        return {}
    
    def update_retriever_k(self, k: int) -> None:
        """Update number of documents returned by retriever"""
        self.retriever_k = k
        self.retriever = self._get_retriever()
        self.chain = self._build_chain()
        logger.info(f"Retriever document count updated to: {k}")

    def _format_docs(self, docs: List[Document]) -> str:
        # Implementation of _format_docs method
        pass

    def is_langsmith_enabled(self):
        return self.langsmith_config is not None


class ConversationalRetrievalChain:
    """Conversational retrieval Q&A chain"""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = None,
        retriever_k: int = 4
    ):
        """
        Initialize conversational retrieval chain
        
        Args:
            vector_store_manager: Vector store manager
            model_name: Model name to use
            retriever_k: Number of documents returned by retriever
        """
        self.vector_store_manager = vector_store_manager
        self.model_name = model_name
        self.retriever_k = retriever_k
        
        # Initialize LLM and prompt manager
        self.llm = LLMFactory.create_llm(model_name)
        self.prompt_manager = PromptTemplateManager()
        self.memory_manager = ConversationMemoryManager()
        
        # Get retriever
        self.retriever = self._get_retriever()
        
        # Build chains
        self.standalone_question_chain = self._build_standalone_question_chain()
        self.qa_chain = self._build_qa_chain()
        
        logger.info("Conversational retrieval Q&A chain initialization completed")
    
    def _get_retriever(self) -> BaseRetriever:
        """Get retriever with Rerank functionality"""
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
        
        if not self.vector_store_manager.vector_store:
            self.vector_store_manager.get_or_create_vector_store()
        
        base_retriever = self.vector_store_manager.get_retriever(k=self.retriever_k)
        
        # Initialize FlashrankRerank
        reranker = FlashrankRerank(model=settings.reranker_model, top_n=3)
        
        # Create compression retriever with contextual compression
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=base_retriever
        )
        
        logger.info("Created compression retriever with CrossEncoder Rerank functionality")
        return compression_retriever
    
    def _build_standalone_question_chain(self) -> Runnable:
        """Build standalone question generation chain"""
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
        """Build Q&A chain"""
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
        Invoke conversational retrieval Q&A chain
        
        Args:
            question: User question
            session_id: Session ID
            **kwargs: Other parameters
        
        Returns:
            Dictionary containing answer and related information
        """
        try:
            logger.info(f"Conversational processing question: {question[:100]}...")
            
            # 1. Generate standalone question
            standalone_question = self.standalone_question_chain.invoke({
                "question": question,
                "session_id": session_id
            })
            
            logger.debug(f"Standalone question: {standalone_question}")
            
            # 2. Use standalone question for Q&A
            answer = self.qa_chain.invoke(standalone_question)
            
            # 3. Get relevant documents
            relevant_docs = self.retriever.invoke(standalone_question)
            
            # 4. Save to memory
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
            
            logger.info("Conversational Q&A processing completed")
            return result
            
        except Exception as e:
            logger.error(f"Conversational Q&A processing failed: {str(e)}")
            return {
                "answer": f"Sorry, an error occurred while processing your question: {str(e)}",
                "question": question,
                "relevant_documents": [],
                "session_id": session_id,
                "error": str(e)
            }

    def _get_standalone_question(self, question: str, chat_history: list) -> str:
        """Generate a standalone question that can be understood without context based on conversation history."""
        if not chat_history:
            return question

        chain = self._get_standalone_question_chain()
        result = chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })
        return result

    def _get_standalone_question_chain(self):
        # 1. Get relevant documents
        relevant_docs = self.retriever.invoke(self.question)

        # 2. Build context
        context = self._format_docs(relevant_docs)

        # 3. Generate standalone question
        standalone_question = self._get_standalone_question(self.question, self.chat_history)

        # 4. Get relevant documents
        relevant_docs = self.retriever.invoke(standalone_question)

        # 5. Build context
        context = self._format_docs(relevant_docs)

        return standalone_question

    def get_relevant_documents(self, question: str) -> List[Document]:
        """Get documents relevant to the question"""
        logger.info(f"Retrieving relevant documents for question: {question[:100]}...")
        return self.retriever.invoke(question)

    def _format_docs(self, docs: List[Document]) -> str:
        # Implementation of _format_docs method
        pass