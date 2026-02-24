from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field
from app.core.constants import DocumentSourceType


class DocumentUploadResponse(BaseModel):
    id: UUID
    user_id: UUID
    chat_session_id: UUID
    source_type: DocumentSourceType
    filename: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
