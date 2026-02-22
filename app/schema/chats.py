from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class ChatSessionResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UpdateChatSessionTitleRequest(BaseModel):
    title: str = Field(..., description="The new title for the chat session")


class ChatMessageResponse(BaseModel):
    id: UUID
    session_id: UUID
    role: str  # TODO: change to literal
    content: str
    created_at: datetime
    ndcg_score: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)
