from typing import List
from uuid import UUID
from sqlalchemy.future import select
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_db
from app.db.models.chat import ChatSession
from app.db.models.user import User
from app.schema.chats import ChatSessionResponse, UpdateChatSessionTitleRequest
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
