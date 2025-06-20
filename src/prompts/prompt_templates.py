from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from typing import List, Dict, Any


class PromptTemplateManager:
    """Prompt template manager"""
    
    @staticmethod
    def get_qa_prompt() -> PromptTemplate:
        """Get basic Q&A prompt template"""
        template = """Answer the question based on the following context information. If there is no relevant content in the context information, please clearly state that the answer cannot be found in the provided documents.

Context Information:
{context}

Question: {question}

Please provide an accurate and detailed answer based on the context information:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    @staticmethod
    def get_chat_qa_prompt() -> ChatPromptTemplate:
        """Get conversational Q&A prompt template (supports chat history)"""
        system_message = """You are a professional document Q&A assistant. Please answer user questions based on the provided context information.

Answer Requirements:
1. Answer based on the provided context information
2. If there is no relevant information in the context, please clearly state so
3. Maintain accuracy and relevance of the answer
4. Consider previous conversation history to provide coherent answers
5. Answer in English

Context Information:
{context}"""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
    
    @staticmethod
    def get_summarization_prompt() -> PromptTemplate:
        """Get document summarization prompt template"""
        template = """Please summarize the following document content and extract the main information and key points:

Document Content:
{text}

Summary:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["text"]
        )
    
    @staticmethod
    def get_standalone_question_prompt() -> PromptTemplate:
        """Get standalone question reconstruction prompt template"""
        template = """Based on the conversation history and the latest user question, rephrase the user question as an independent, complete question that can be understood without referring to the conversation history.

Conversation History:
{chat_history}

Latest Question: {question}

Standalone Question:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["chat_history", "question"]
        )
    
    @staticmethod
    def get_multi_query_prompt() -> PromptTemplate:
        """Get multi-query generation prompt template"""
        template = """You are an AI language model assistant. Your task is to generate 3 different versions of the user question to retrieve relevant documents from a vector database. By generating multiple perspectives of the question, your goal is to help the user overcome some of the limitations of distance-based similarity search. Please separate these alternative questions with newlines.

Original Question: {question}"""
        
        return PromptTemplate(
            template=template,
            input_variables=["question"]
        )
    
    @staticmethod
    def get_custom_prompt(
        template: str,
        input_variables: List[str]
    ) -> PromptTemplate:
        """Create custom prompt template"""
        return PromptTemplate(
            template=template,
            input_variables=input_variables
        )
    
    @staticmethod
    def get_context_compression_prompt() -> PromptTemplate:
        """Get context compression prompt template"""
        template = """Based on the user's question, extract the most relevant information from the following document fragments:

Question: {question}

Document Fragments:
{context}

Please return only information directly relevant to the question, removing irrelevant content:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["question", "context"]
        )


class PromptFormatter:
    """Prompt formatting tool"""
    
    @staticmethod
    def format_documents(docs: List[Any]) -> str:
        """Format document list as string"""
        if not docs:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            source = doc.metadata.get('source_file', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
            formatted_docs.append(f"Document {i} (Source: {source}):\n{content}")
        
        return "\n\n".join(formatted_docs)
    
    @staticmethod
    def format_chat_history(messages: List[BaseMessage]) -> str:
        """Format chat history"""
        if not messages:
            return "No conversation history"
        
        formatted_history = []
        for message in messages:
            if hasattr(message, 'type'):
                role = "User" if message.type == "human" else "Assistant"
                formatted_history.append(f"{role}: {message.content}")
            else:
                formatted_history.append(str(message))
        
        return "\n".join(formatted_history)
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 4000) -> str:
        """Truncate text to specified length"""
        if len(text) <= max_length:
            return text
        
        return text[:max_length] + "...[Text truncated]" 