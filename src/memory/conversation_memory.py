from typing import List, Dict, Any
import logging
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """对话记忆管理器，专注于管理聊天历史记录"""
    
    def __init__(self):
        """
        初始化对话记忆管理器
        """
        self.session_histories: Dict[str, ChatMessageHistory] = {}
    
    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        """获取或创建会话历史"""
        if session_id not in self.session_histories:
            self.session_histories[session_id] = ChatMessageHistory()
            logger.info(f"为会话 {session_id} 创建新的聊天记录")
        return self.session_histories[session_id]

    def add_user_message(self, message: str, session_id: str = "default") -> None:
        """添加用户消息"""
        try:
            history = self._get_session_history(session_id)
            history.add_user_message(message)
            logger.debug(f"向会话 {session_id} 添加用户消息: {message[:50]}...")
        except Exception as e:
            logger.error(f"向会话 {session_id} 添加用户消息失败: {str(e)}")
    
    def add_ai_message(self, message: str, session_id: str = "default") -> None:
        """添加AI消息"""
        try:
            history = self._get_session_history(session_id)
            history.add_ai_message(message)
            logger.debug(f"向会话 {session_id} 添加AI消息: {message[:50]}...")
        except Exception as e:
            logger.error(f"向会话 {session_id} 添加AI消息失败: {str(e)}")
    
    def add_message_pair(self, user_message: str, ai_message: str, session_id: str = "default") -> None:
        """添加一对对话消息"""
        self.add_user_message(user_message, session_id)
        self.add_ai_message(ai_message, session_id)
    
    def get_chat_history(self, session_id: str = "default") -> List[BaseMessage]:
        """获取聊天历史"""
        try:
            return self._get_session_history(session_id).messages
        except Exception as e:
            logger.error(f"获取会话 {session_id} 的聊天历史失败: {str(e)}")
            return []
    
    def clear_memory(self, session_id: str = "default") -> None:
        """清空指定会话的记忆"""
        try:
            if session_id in self.session_histories:
                self.session_histories[session_id].clear()
                logger.info(f"已清空会话 {session_id} 的记忆")
        except Exception as e:
            logger.error(f"清空会话 {session_id} 的记忆失败: {str(e)}")
    
    def delete_session(self, session_id: str) -> None:
        """删除会话及其历史记录"""
        if session_id in self.session_histories:
            del self.session_histories[session_id]
            logger.info(f"已删除会话: {session_id}")
    
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
                "total_characters": total_chars
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
            for message in reversed(chat_history): # 从最新消息开始
                role = "用户" if isinstance(message, HumanMessage) else "助手"
                formatted_messages.append(f"{role}: {message.content}")
            
            result = "\n".join(reversed(formatted_messages)) # 恢复原始顺序
            
            if len(result) > max_length:
                result = "...[历史记录已截断]\n" + result[-max_length:]
            
            return result
        except Exception as e:
            logger.error(f"格式化聊天历史失败: {str(e)}")
            return ""


class SessionManager:
    """会话管理器，提供对ConversationMemoryManager的单例访问"""
    
    _instance = None
    _memory_manager: ConversationMemoryManager = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._memory_manager = ConversationMemoryManager()
            logger.info("SessionManager已初始化，创建了ConversationMemoryManager单例")
        return cls._instance

    def get_memory_manager(self) -> ConversationMemoryManager:
        """获取ConversationMemoryManager的单例实例"""
        return self._memory_manager
    
    def get_history(self, session_id: str) -> List[BaseMessage]:
        """获取指定会话的聊天历史"""
        return self._memory_manager.get_chat_history(session_id)

    def add_message_pair(self, session_id: str, user_message: str, ai_message: str):
        """向指定会话添加一对消息"""
        self._memory_manager.add_message_pair(user_message, ai_message, session_id)

    def delete_session(self, session_id: str):
        """删除指定会d话"""
        self._memory_manager.delete_session(session_id)
    
    def get_all_sessions(self) -> List[str]:
        """获取所有会话ID"""
        return self._memory_manager.get_sessions()
    
    def clear_all_sessions(self) -> None:
        """清空所有会话"""
        all_sessions = self._memory_manager.get_sessions()
        for session_id in all_sessions:
            self._memory_manager.clear_memory(session_id)
        logger.info("已清空所有会话")