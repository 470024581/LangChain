from typing import List, Dict, Any
import logging
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """Conversation memory manager, focused on managing chat history"""
    
    def __init__(self):
        """
        Initialize conversation memory manager
        """
        self.session_histories: Dict[str, ChatMessageHistory] = {}
    
    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        """Get or create session history"""
        if session_id not in self.session_histories:
            self.session_histories[session_id] = ChatMessageHistory()
            logger.info(f"Created new chat history for session {session_id}")
        return self.session_histories[session_id]

    def add_user_message(self, message: str, session_id: str = "default") -> None:
        """Add user message"""
        try:
            history = self._get_session_history(session_id)
            history.add_user_message(message)
            logger.debug(f"Added user message to session {session_id}: {message[:50]}...")
        except Exception as e:
            logger.error(f"Failed to add user message to session {session_id}: {str(e)}")
    
    def add_ai_message(self, message: str, session_id: str = "default") -> None:
        """Add AI message"""
        try:
            history = self._get_session_history(session_id)
            history.add_ai_message(message)
            logger.debug(f"Added AI message to session {session_id}: {message[:50]}...")
        except Exception as e:
            logger.error(f"Failed to add AI message to session {session_id}: {str(e)}")
    
    def add_message_pair(self, user_message: str, ai_message: str, session_id: str = "default") -> None:
        """Add a pair of conversation messages"""
        self.add_user_message(user_message, session_id)
        self.add_ai_message(ai_message, session_id)
    
    def get_chat_history(self, session_id: str = "default") -> List[BaseMessage]:
        """Get chat history"""
        try:
            return self._get_session_history(session_id).messages
        except Exception as e:
            logger.error(f"Failed to get chat history for session {session_id}: {str(e)}")
            return []
    
    def clear_memory(self, session_id: str = "default") -> None:
        """Clear memory for specified session"""
        try:
            if session_id in self.session_histories:
                self.session_histories[session_id].clear()
                logger.info(f"Cleared memory for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to clear memory for session {session_id}: {str(e)}")
    
    def delete_session(self, session_id: str) -> None:
        """Delete session and its history"""
        if session_id in self.session_histories:
            del self.session_histories[session_id]
            logger.info(f"Deleted session: {session_id}")
    
    def get_sessions(self) -> List[str]:
        """Get all session IDs"""
        return list(self.session_histories.keys())
    
    def get_memory_stats(self, session_id: str = "default") -> Dict[str, Any]:
        """Get memory statistics"""
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
            logger.error(f"Failed to get memory statistics: {str(e)}")
            return {}
    
    def format_chat_history_for_prompt(self, session_id: str = "default", max_length: int = 2000) -> str:
        """Format chat history for prompt"""
        try:
            chat_history = self.get_chat_history(session_id)
            if not chat_history:
                return ""
            
            formatted_messages = []
            for message in reversed(chat_history): # Start from newest message
                role = "User" if isinstance(message, HumanMessage) else "Assistant"
                formatted_messages.append(f"{role}: {message.content}")
            
            result = "\n".join(reversed(formatted_messages)) # Restore original order
            
            if len(result) > max_length:
                result = "...[History truncated]\n" + result[-max_length:]
            
            return result
        except Exception as e:
            logger.error(f"Failed to format chat history: {str(e)}")
            return ""


class SessionManager:
    """Session manager, providing singleton access to ConversationMemoryManager"""
    
    _instance = None
    _memory_manager: ConversationMemoryManager = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._memory_manager = ConversationMemoryManager()
            logger.info("SessionManager initialized, created ConversationMemoryManager singleton")
        return cls._instance

    def get_memory_manager(self) -> ConversationMemoryManager:
        """Get singleton instance of ConversationMemoryManager"""
        return self._memory_manager
    
    def get_history(self, session_id: str) -> List[BaseMessage]:
        """Get chat history for specified session"""
        return self._memory_manager.get_chat_history(session_id)

    def add_message_pair(self, session_id: str, user_message: str, ai_message: str):
        """Add a pair of messages to specified session"""
        self._memory_manager.add_message_pair(user_message, ai_message, session_id)

    def delete_session(self, session_id: str):
        """Delete specified session"""
        self._memory_manager.delete_session(session_id)
    
    def get_all_sessions(self) -> List[str]:
        """Get all session IDs"""
        return self._memory_manager.get_sessions()
    
    def clear_all_sessions(self) -> None:
        """Clear all sessions"""
        all_sessions = self._memory_manager.get_sessions()
        for session_id in all_sessions:
            self._memory_manager.clear_memory(session_id)
        logger.info("Cleared all sessions")