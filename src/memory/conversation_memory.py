from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """对话记忆管理器"""
    
    def __init__(self, memory_type: str = "buffer_window", max_token_limit: int = 2000, k: int = 5):
        """
        初始化对话记忆管理器
        
        Args:
            memory_type: 记忆类型 ("buffer", "buffer_window", "summary")
            max_token_limit: 最大token限制
            k: 窗口大小（仅对buffer_window有效）
        """
        self.memory_type = memory_type
        self.max_token_limit = max_token_limit
        self.k = k
        self.memory = self._create_memory()
        self.session_histories: Dict[str, ChatMessageHistory] = {}
    
    def _create_memory(self):
        """创建记忆实例"""
        if self.memory_type == "buffer":
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=self.max_token_limit
            )
        elif self.memory_type == "buffer_window":
            return ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=self.k
            )
        else:
            # 默认使用窗口记忆
            return ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=self.k
            )
    
    def add_user_message(self, message: str, session_id: str = "default") -> None:
        """添加用户消息"""
        try:
            if session_id in self.session_histories:
                self.session_histories[session_id].add_user_message(message)
            else:
                self.memory.chat_memory.add_user_message(message)
            logger.debug(f"添加用户消息: {message[:50]}...")
        except Exception as e:
            logger.error(f"添加用户消息失败: {str(e)}")
    
    def add_ai_message(self, message: str, session_id: str = "default") -> None:
        """添加AI消息"""
        try:
            if session_id in self.session_histories:
                self.session_histories[session_id].add_ai_message(message)
            else:
                self.memory.chat_memory.add_ai_message(message)
            logger.debug(f"添加AI消息: {message[:50]}...")
        except Exception as e:
            logger.error(f"添加AI消息失败: {str(e)}")
    
    def add_message_pair(self, user_message: str, ai_message: str, session_id: str = "default") -> None:
        """添加一对对话消息"""
        self.add_user_message(user_message, session_id)
        self.add_ai_message(ai_message, session_id)
    
    def get_memory_variables(self, session_id: str = "default") -> Dict[str, Any]:
        """获取记忆变量"""
        try:
            if session_id in self.session_histories:
                return {"chat_history": self.session_histories[session_id].messages}
            else:
                return self.memory.load_memory_variables({})
        except Exception as e:
            logger.error(f"获取记忆变量失败: {str(e)}")
            return {"chat_history": []}
    
    def get_chat_history(self, session_id: str = "default") -> List[BaseMessage]:
        """获取聊天历史"""
        try:
            if session_id in self.session_histories:
                return self.session_histories[session_id].messages
            else:
                return self.memory.chat_memory.messages
        except Exception as e:
            logger.error(f"获取聊天历史失败: {str(e)}")
            return []
    
    def clear_memory(self, session_id: str = "default") -> None:
        """清空记忆"""
        try:
            if session_id in self.session_histories:
                self.session_histories[session_id].clear()
                logger.info(f"已清空会话 {session_id} 的记忆")
            else:
                self.memory.clear()
                logger.info("已清空默认记忆")
        except Exception as e:
            logger.error(f"清空记忆失败: {str(e)}")
    
    def create_session(self, session_id: str) -> None:
        """创建新的会话"""
        if session_id not in self.session_histories:
            self.session_histories[session_id] = ChatMessageHistory()
            logger.info(f"创建新会话: {session_id}")
    
    def delete_session(self, session_id: str) -> None:
        """删除会话"""
        if session_id in self.session_histories:
            del self.session_histories[session_id]
            logger.info(f"删除会话: {session_id}")
    
    def get_sessions(self) -> List[str]:
        """获取所有会话ID"""
        return list(self.session_histories.keys())
    
    def get_memory_stats(self, session_id: str = "default") -> Dict[str, Any]:
        """获取记忆统计信息"""
        try:
            chat_history = self.get_chat_history(session_id)
            message_count = len(chat_history)
            
            user_messages = sum(1 for msg in chat_history if isinstance(msg, HumanMessage))
            ai_messages = sum(1 for msg in chat_history if isinstance(msg, AIMessage))
            
            total_chars = sum(len(msg.content) for msg in chat_history)
            
            return {
                "session_id": session_id,
                "total_messages": message_count,
                "user_messages": user_messages,
                "ai_messages": ai_messages,
                "total_characters": total_chars,
                "memory_type": self.memory_type
            }
        except Exception as e:
            logger.error(f"获取记忆统计信息失败: {str(e)}")
            return {}
    
    def format_chat_history_for_prompt(self, session_id: str = "default", max_length: int = 2000) -> str:
        """格式化聊天历史用于提示词"""
        try:
            chat_history = self.get_chat_history(session_id)
            if not chat_history:
                return ""
            
            formatted_messages = []
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    formatted_messages.append(f"用户: {message.content}")
                elif isinstance(message, AIMessage):
                    formatted_messages.append(f"助手: {message.content}")
            
            result = "\n".join(formatted_messages)
            
            # 如果太长，截断到指定长度
            if len(result) > max_length:
                result = result[-max_length:] + "...[历史记录已截断]"
            
            return result
        except Exception as e:
            logger.error(f"格式化聊天历史失败: {str(e)}")
            return ""


class SessionManager:
    """会话管理器"""
    
    def __init__(self):
        self.memory_managers: Dict[str, ConversationMemoryManager] = {}
    
    def get_or_create_memory_manager(
        self, 
        session_id: str,
        memory_type: str = "buffer_window",
        **kwargs
    ) -> ConversationMemoryManager:
        """获取或创建记忆管理器"""
        if session_id not in self.memory_managers:
            self.memory_managers[session_id] = ConversationMemoryManager(
                memory_type=memory_type,
                **kwargs
            )
            logger.info(f"为会话 {session_id} 创建记忆管理器")
        
        return self.memory_managers[session_id]
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self.memory_managers:
            del self.memory_managers[session_id]
            logger.info(f"删除会话: {session_id}")
            return True
        return False
    
    def get_all_sessions(self) -> List[str]:
        """获取所有会话ID"""
        return list(self.memory_managers.keys())
    
    def clear_all_sessions(self) -> None:
        """清空所有会话"""
        self.memory_managers.clear()
        logger.info("已清空所有会话") 