from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from ..db.models import ChatSession, ChatMessage

class MessageHistoryManager:
    """Quản lý lịch sử chat từ database"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_or_create_session(self, user_id: int, session_id: int | None = None) -> int:
        """
        Lấy hoặc tạo session mới cho user
        
        Args:
            user_id: ID của user
            session_id: ID session cũ (nếu có)
            
        Returns:
            session_id: ID của session (mới hoặc cũ)
        """
        if session_id:
            # Kiểm tra session có tồn tại và thuộc user này không
            result = await self.db.execute(
                select(ChatSession).where(
                    ChatSession.id == session_id,
                    ChatSession.user_id == user_id
                )
            )
            session = result.scalar_one_or_none()
            if session:
                return session.id
        
        # Tạo session mới
        new_session = ChatSession(user_id=user_id)
        self.db.add(new_session)
        await self.db.commit()
        await self.db.refresh(new_session)
        return new_session.id
    
    async def get_messages(self, session_id: int, limit: int = 10) -> list[BaseMessage]:
        """
        Lấy N tin nhắn gần nhất từ session
        
        Args:
            session_id: ID của session
            limit: Số lượng tin nhắn tối đa
            
        Returns:
            List các LangChain messages (HumanMessage, AIMessage)
        """
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        messages = result.scalars().all()
        
        # Convert sang LangChain messages
        langchain_messages = []
        for msg in reversed(messages):  # Đảo ngược để đúng thứ tự thời gian
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
        
        return langchain_messages
    
    async def add_message(self, session_id: int, role: str, content: str) -> None:
        """
        Thêm message mới vào session
        
        Args:
            session_id: ID của session
            role: 'user' hoặc 'assistant'
            content: Nội dung tin nhắn
        """
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content
        )
        self.db.add(message)
        await self.db.commit()
    
    async def clear_session(self, session_id: int) -> None:
        """Xóa toàn bộ messages trong một session"""
        await self.db.execute(
            select(ChatMessage).where(ChatMessage.session_id == session_id)
        )
        await self.db.commit()
