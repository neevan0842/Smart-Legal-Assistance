from datetime import datetime
from typing import List, Literal, Optional
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


class Risk(BaseModel):
    clause_reference: Optional[str]
    risk_level: Literal["Low", "Medium", "High"]
    issue: str
    recommendation: str

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class DocumentAnalysis(BaseModel):
    summary: Optional[str]
    risks: Optional[List[Risk]]
    suggestions: Optional[List[str]]
    missing_clauses: Optional[List[str]]

    model_config = ConfigDict(from_attributes=True, extra="forbid")
