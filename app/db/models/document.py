import uuid
from sqlalchemy import Enum, String, TIMESTAMP, func, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from app.core.constants import DocumentSourceType
from app.db.base import Base


class MessageDocument(Base):
    __tablename__ = "message_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_messages.id"), nullable=False
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False
    )

    document = relationship(
        "Document", back_populates="message_documents", cascade="all, delete"
    )
    message = relationship(
        "ChatMessage", back_populates="message_documents", cascade="all, delete"
    )

    def __repr__(self) -> str:
        return f"<MessageDocument(id={self.id}, message_id={self.message_id}, document_id={self.document_id})>"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    chat_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=True
    )
    source_type: Mapped[DocumentSourceType] = mapped_column(
        Enum(DocumentSourceType, native_enum=True), nullable=False
    )
    filename: Mapped[str] = mapped_column(String, nullable=False)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )

    user = relationship("User", back_populates="documents")
    chat_session = relationship("ChatSession", back_populates="documents")
    message_documents = relationship(
        "MessageDocument", back_populates="document", cascade="all, delete"
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.filename}, user_id={self.user_id}, storage_path={self.storage_path})>"
