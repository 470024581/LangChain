"""
Agent-based RAG (Retrieval-Augmented Generation) Implementation

This module implements a RAG system using LangChain's built-in Agent,
maintaining the same logic and interface as the existing Chain version.
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
    Agent-based Document Q&A System
    
    This class implements the same functionality as DocumentQAChain, but uses LangChain's Agent framework.
    """
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = None,
        use_memory: bool = True,
        retriever_k: int = 4
    ):
        """
        Initialize Agent
        
        Args:
            vector_store_manager: Vector store manager
            model_name: Model name
            use_memory: Whether to use memory
            retriever_k: Number of documents returned by retriever
        """
        self.vector_store_manager = vector_store_manager
        self.model_name = model_name
        self.use_memory = use_memory
        self.retriever_k = retriever_k
        
        # Initialize components
        self.llm = LLMFactory.create_llm(model_name)
        self.prompt_manager = PromptTemplateManager()
        self.memory_manager = ConversationMemoryManager() if use_memory else None
        
        # Initialize retriever
        self.retriever = self._get_retriever()
        
        # Build Agent
        self.agent_executor = self._build_agent()
        
        # Configure LangSmith
        self._configure_langsmith()
        
        logger.info("DocumentQAAgent initialization completed")
    
    def _get_retriever(self) -> BaseRetriever:
        """Get retriever, prioritize Rerank functionality, fallback to basic retriever on failure"""
        if not self.vector_store_manager.vector_store:
            self.vector_store_manager.get_or_create_vector_store()
        
        base_retriever = self.vector_store_manager.get_retriever(k=self.retriever_k)
        
        # Try to use FlashrankRerank, fallback to basic retriever on failure
        try:
            from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
            
            # Initialize FlashrankRerank
            reranker = FlashrankRerank(model=settings.reranker_model, top_n=3)
            
            # Create retriever with contextual compression
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=reranker, 
                base_retriever=base_retriever
            )
            
            logger.info("Created compression retriever with FlashrankRerank functionality")
            return compression_retriever
            
        except Exception as e:
            logger.warning(f"FlashrankRerank initialization failed: {e}")
            logger.info("Falling back to basic retriever")
            return base_retriever
    
    def _create_tools(self) -> List[Tool]:
        """Create Agent tools"""
        tools = [
            Tool(
                name="document_retrieval",
                description="Search and retrieve document content related to the question. Used to answer questions that require document-based knowledge.",
                func=self.retriever.invoke
            )
        ]
        return tools
    
    def _build_agent(self) -> AgentExecutor:
        """Build Agent executor"""
        tools = self._create_tools()
        
        # Use simple system message instead of complex prompt template
        system_message = """You are an intelligent document Q&A assistant. You need to answer user questions based on retrieved document content.

Workflow:
1. Use the document_retrieval tool once to search for relevant documents.
2. Analyze the retrieved document content and generate the final answer based on it.

**Very Important**: If you decide to use tools, your response **must** only contain the tool call JSON, without any other text. Once you get the information returned by the tool, then organize the language to answer the question. If there is no relevant information in the documents, please state clearly.
"""

        # Create prompt template, let LangChain automatically handle the tool part
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
        
        # Create tool-calling agent
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        # Create Agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True,
            return_intermediate_steps=True, # Ensure intermediate steps are returned
        )
        
        return agent_executor
    
    def _configure_langsmith(self):
        """Configure LangSmith tracing"""
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
        """Prepare Agent input"""
        input_data = {"input": question}
        
        if self.use_memory and self.memory_manager:
            # Add chat history
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
        Invoke Q&A Agent
        
        Args:
            question: User question
            session_id: Session ID (if using memory)
            **kwargs: Other parameters
        
        Returns:
            Dictionary containing answer and related information
        """
        try:
            logger.info(f"Agent processing question: {question[:100]}...")
            
            # Prepare input
            input_data = self._prepare_input(question, session_id)
            
            # Invoke Agent (use config if LangSmith is enabled)
            if self.langsmith_config:
                result = self.agent_executor.invoke(input_data, config=self.langsmith_config)
            else:
                result = self.agent_executor.invoke(input_data)
            
            answer = result.get("output", "")
            
            # Extract relevant documents from intermediate steps
            relevant_docs = []
            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    if action.tool == "document_retrieval" and isinstance(observation, list):
                        relevant_docs.extend(doc for doc in observation if isinstance(doc, Document))

            # If no documents found from intermediate steps, perform fallback retrieval
            if not relevant_docs and question:
                logger.warning("Unable to extract documents from Agent intermediate steps, performing fallback retrieval.")
                relevant_docs = self.retriever.invoke(question)
            
            # Save to memory (if enabled)
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
            
            logger.info("Agent Q&A processing completed")
            return response
            
        except Exception as e:
            logger.error(f"Agent Q&A processing failed: {str(e)}")
            return {
                "answer": f"Sorry, an error occurred while processing your question: {str(e)}",
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
        Asynchronously invoke Q&A Agent
        
        Args:
            question: User question
            session_id: Session ID
            **kwargs: Other parameters
        
        Returns:
            Dictionary containing answer and related information
        """
        try:
            logger.info(f"Agent async processing question: {question[:100]}...")
            
            # Prepare input
            input_data = self._prepare_input(question, session_id)
            
            # Async invoke Agent
            if self.langsmith_config:
                result = await self.agent_executor.ainvoke(input_data, config=self.langsmith_config)
            else:
                result = await self.agent_executor.ainvoke(input_data)
            
            answer = result.get("output", "")
            
            # Async extract documents from intermediate steps
            relevant_docs = []
            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    if action.tool == "document_retrieval" and isinstance(observation, list):
                        relevant_docs.extend(doc for doc in observation if isinstance(doc, Document))
            
            if not relevant_docs and question:
                logger.warning("Unable to extract documents from Agent intermediate steps, performing fallback retrieval.")
                relevant_docs = await self.retriever.ainvoke(question)

            # Save to memory (if enabled)
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
            
            logger.info("Agent async Q&A processing completed")
            return response
            
        except Exception as e:
            logger.error(f"Agent async Q&A processing failed: {str(e)}")
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
        Stream invoke Q&A Agent
        
        Args:
            question: User question
            session_id: Session ID
            **kwargs: Other parameters
        
        Yields:
            Streaming output during Agent execution
        """
        try:
            logger.info(f"Agent streaming processing question: {question[:100]}...")
            
            # Prepare input
            input_data = self._prepare_input(question, session_id)
            
            full_answer = ""
            # Stream invoke Agent
            for chunk in self.agent_executor.stream(input_data):
                if "output" in chunk:
                    content = chunk["output"]
                    full_answer += content
                    yield content
                elif "intermediate_step" in chunk:
                    # Can also output intermediate steps
                    step = chunk["intermediate_step"]
                    yield f"\n[Agent thinking]: {step}\n"
            
            # Save to memory (if enabled)
            if self.use_memory and self.memory_manager:
                self.memory_manager.add_message_pair(
                    user_message=question,
                    ai_message=full_answer,
                    session_id=session_id
                )
            
        except Exception as e:
            logger.error(f"Agent streaming Q&A processing failed: {str(e)}")
            yield f"Sorry, an error occurred while processing your question: {str(e)}"
    
    def get_relevant_documents(self, question: str) -> List[Document]:
        """Get documents relevant to the question"""
        logger.info(f"Agent retrieving relevant documents for question: {question[:100]}...")
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
        else:
            return {"use_memory": False}
    
    def update_retriever_k(self, k: int) -> None:
        """Update number of documents returned by retriever"""
        self.retriever_k = k
        self.retriever = self._get_retriever()
        # Rebuild Agent
        self.agent_executor = self._build_agent()
        logger.info(f"Retriever document count updated to: {k}")


class ConversationalRetrievalAgent:
    """
    Agent-based Conversational Retrieval Chain
    
    Implements the same functionality as ConversationalRetrievalChain, using Agent architecture.
    """
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = None,
        retriever_k: int = 4
    ):
        """
        Initialize Conversational Retrieval Agent
        
        Args:
            vector_store_manager: Vector store manager
            model_name: Model name
            retriever_k: Number of documents returned by retriever
        """
        self.vector_store_manager = vector_store_manager
        self.model_name = model_name
        self.retriever_k = retriever_k
        
        # Initialize components
        self.llm = LLMFactory.create_llm(model_name)
        self.prompt_manager = PromptTemplateManager()
        
        # Initialize retriever
        self.retriever = self._get_retriever()
        
        # Build Agent
        self.agent_executor = self._build_agent()
        
        logger.info("ConversationalRetrievalAgent initialization completed")
    
    def _get_retriever(self) -> BaseRetriever:
        """Get retriever with Rerank functionality"""
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
        
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
    
    def _create_tools(self) -> List[Tool]:
        """Create Agent tools"""
        tools = [
            Tool(
                name="document_retrieval",
                description="Search and retrieve document content related to the question.",
                func=self.retriever.invoke,
            )
        ]
        return tools
    
    def _generate_standalone_question(self, input_with_history: str) -> str:
        """Generate standalone question"""
        # More complex logic can be implemented here to extract standalone question from conversation history
        # Simple implementation: directly return the last question
        lines = input_with_history.strip().split('\n')
        return lines[-1] if lines else input_with_history
    
    def _build_agent(self) -> AgentExecutor:
        """Build Agent executor"""
        tools = self._create_tools()
        
        system_message = """You are a conversational document retrieval assistant. You need to:

1. Analyze the user's question and conversation history, and if necessary, rephrase the question for better retrieval.
2. Use the document_retrieval tool to retrieve relevant documents.
3. Answer questions based on retrieval results and conversation context.

**Very Important**: If you decide to use tools, your response **must** only contain the tool call JSON, without any other text. Once you get the information returned by the tool, then organize the language to answer the question. If there is no relevant information in the documents, please state clearly."""

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
        Invoke Conversational Retrieval Agent
        
        Args:
            question: User question
            chat_history: Chat history
            **kwargs: Other parameters
        
        Returns:
            Dictionary containing answer and related information
        """
        try:
            logger.info(f"Conversational Agent processing question: {question[:100]}...")
            
            # Prepare input, including chat history
            if chat_history:
                history_str = "\n".join([f"Human: {h}\nAssistant: {a}" for h, a in chat_history])
                input_text = f"Chat history:\n{history_str}\n\nCurrent question: {question}"
            else:
                input_text = question
            
            input_data = {"input": input_text}
            
            # Invoke Agent
            result = self.agent_executor.invoke(input_data)
            answer = result.get("output", "")
            
            # Extract relevant documents from intermediate steps
            relevant_docs = []
            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    if action.tool == "document_retrieval" and isinstance(observation, list):
                        relevant_docs.extend(doc for doc in observation if isinstance(doc, Document))

            if not relevant_docs and question:
                logger.warning("Unable to extract documents from Agent intermediate steps, performing fallback retrieval.")
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
            
            logger.info("Conversational Agent Q&A processing completed")
            return response
            
        except Exception as e:
            logger.error(f"Conversational Agent Q&A processing failed: {str(e)}")
            return {
                "answer": f"Sorry, an error occurred while processing your question: {str(e)}",
                "question": question,
                "relevant_documents": [],
                "chat_history": chat_history,
                "error": str(e)
            }