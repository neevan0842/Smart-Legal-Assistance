from typing import List
from uuid import UUID
from sqlalchemy.future import select
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_db
from app.db.models.chat import ChatMessage, ChatSession
from app.db.models.user import User
from app.schema.chats import (
    ChatMessageResponse,
    ChatSessionResponse,
    UpdateChatSessionTitleRequest,
)
from app.service.chat import get_chat_messages_with_score
from app.service.users import get_current_active_user


router = APIRouter(prefix="/chats", tags=["chats"])


@router.get("/", response_model=List[ChatSessionResponse])
async def get_all_user_chats(
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Endpoint to retrieve the list of chats for the authenticated user."""
    stmt = (
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.created_at.desc())
    )
    result = await db.execute(stmt)
    chat_sessions = result.scalars().all()
    return chat_sessions


@router.post(
    "/", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED
)
async def create_new_chat_session(
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Endpoint to create a new chat session for the authenticated user."""
    new_chat = ChatSession(user_id=user.id)
    db.add(new_chat)
    await db.commit()
    await db.refresh(new_chat)
    new_message = ChatMessage(
        session_id=new_chat.id,
        role="system",  # TODO:change to literal
        content="Hello! How can I assist you today?",
    )
    db.add(new_message)
    await db.commit()
    return new_chat


@router.patch("/{chat_id}", response_model=ChatSessionResponse)
async def update_chat_session_title(
    chat_id: UUID,
    payload: UpdateChatSessionTitleRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Endpoint to update the title of a chat session for the authenticated user."""
    stmt = select(ChatSession).where(
        ChatSession.id == chat_id, ChatSession.user_id == user.id
    )
    result = await db.execute(stmt)
    chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found"
        )
    chat_session.title = payload.title
    await db.commit()
    await db.refresh(chat_session)
    return chat_session


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    chat_id: UUID,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Endpoint to delete a chat session for the authenticated user."""
    stmt = select(ChatSession).where(
        ChatSession.id == chat_id, ChatSession.user_id == user.id
    )
    result = await db.execute(stmt)
    chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found"
        )
    await db.delete(chat_session)
    await db.commit()
    return None


@router.get("/{chat_id}/messages", response_model=List[ChatMessageResponse])
async def get_chat_messages_by_chat_id(
    chat_id: UUID,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Endpoint to retrieve the list of messages for a specific chat session of the authenticated user."""
    # Verify that the chat session exists and belongs to the user
    stmt = select(ChatSession).where(
        ChatSession.id == chat_id, ChatSession.user_id == user.id
    )
    result = await db.execute(stmt)
    chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found"
        )

    # Retrieve messages for the chat session
    messages = await get_chat_messages_with_score(chat_id=chat_id, db=db)
    return messages
