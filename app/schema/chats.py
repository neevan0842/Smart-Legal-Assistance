from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field
from app.core.constants import ChatRole, DocumentSourceType


class ChatSessionResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UpdateChatSessionTitleRequest(BaseModel):
    title: str = Field(..., description="The new title for the chat session")


class DocumentResponse(BaseModel):
    id: UUID
    source_type: DocumentSourceType
    filename: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatMessageResponse(BaseModel):
    id: UUID
    session_id: UUID
    role: ChatRole
    content: str
    created_at: datetime
    ndcg_score: Optional[float] = None
    documents: List[DocumentResponse] = []

    model_config = ConfigDict(from_attributes=True)


class ChatMessageAIResponse(ChatMessageResponse):
    context: List[str] = []
