from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from typing import List, Dict, Any


class PromptTemplateManager:
    """提示词模板管理器"""
    
    @staticmethod
    def get_qa_prompt() -> PromptTemplate:
        """获取基础问答提示词模板"""
        template = """根据以下上下文信息回答问题。如果上下文信息中没有相关内容，请明确说明无法从提供的文档中找到答案。

上下文信息:
{context}

问题: {question}

请基于上下文信息提供准确、详细的回答:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    @staticmethod
    def get_chat_qa_prompt() -> ChatPromptTemplate:
        """获取对话式问答提示词模板（支持历史对话）"""
        system_message = """你是一个专业的文档问答助手。请根据提供的上下文信息回答用户的问题。

回答要求：
1. 基于提供的上下文信息进行回答
2. 如果上下文中没有相关信息，请明确说明
3. 保持回答的准确性和相关性
4. 考虑之前的对话历史来提供连贯的回答
5. 用中文回答

上下文信息:
{context}"""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
    
    @staticmethod
    def get_summarization_prompt() -> PromptTemplate:
        """获取文档摘要提示词模板"""
        template = """请对以下文档内容进行摘要，提取主要信息和关键点：

文档内容:
{text}

摘要:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["text"]
        )
    
    @staticmethod
    def get_standalone_question_prompt() -> PromptTemplate:
        """获取独立问题重构提示词模板"""
        template = """根据对话历史和最新的用户问题，将用户问题重新表述为一个独立的、完整的问题，
不需要参考对话历史就能理解的问题。

对话历史:
{chat_history}

最新问题: {question}

独立问题:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["chat_history", "question"]
        )
    
    @staticmethod
    def get_multi_query_prompt() -> PromptTemplate:
        """获取多查询生成提示词模板"""
        template = """你是一个AI语言模型助手。你的任务是生成3个不同版本的用户问题，
以便从向量数据库中检索相关文档。通过生成多个角度的问题，
你的目标是帮助用户克服基于距离的相似性搜索的一些限制。
请用换行符分隔这些备选问题。

原始问题: {question}"""
        
        return PromptTemplate(
            template=template,
            input_variables=["question"]
        )
    
    @staticmethod
    def get_custom_prompt(
        template: str,
        input_variables: List[str]
    ) -> PromptTemplate:
        """创建自定义提示词模板"""
        return PromptTemplate(
            template=template,
            input_variables=input_variables
        )
    
    @staticmethod
    def get_context_compression_prompt() -> PromptTemplate:
        """获取上下文压缩提示词模板"""
        template = """根据用户的问题，从以下文档片段中提取最相关的信息：

问题: {question}

文档片段:
{context}

请只返回与问题直接相关的信息，去除无关内容:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["question", "context"]
        )


class PromptFormatter:
    """提示词格式化工具"""
    
    @staticmethod
    def format_documents(docs: List[Any]) -> str:
        """格式化文档列表为字符串"""
        if not docs:
            return "没有找到相关文档。"
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            source = doc.metadata.get('source_file', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
            formatted_docs.append(f"文档 {i} (来源: {source}):\n{content}")
        
        return "\n\n".join(formatted_docs)
    
    @staticmethod
    def format_chat_history(messages: List[BaseMessage]) -> str:
        """格式化聊天历史"""
        if not messages:
            return "无对话历史"
        
        formatted_history = []
        for message in messages:
            if hasattr(message, 'type'):
                role = "用户" if message.type == "human" else "助手"
                formatted_history.append(f"{role}: {message.content}")
            else:
                formatted_history.append(str(message))
        
        return "\n".join(formatted_history)
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 4000) -> str:
        """截断文本到指定长度"""
        if len(text) <= max_length:
            return text
        
        return text[:max_length] + "...[文本已截断]" 