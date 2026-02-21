import uuid
from sqlalchemy import TIMESTAMP, Float, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from app.db.base import Base


class MessageEvaluation(Base):
    __tablename__ = "message_evaluations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    chat_message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    ndcg_score: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )

    message = relationship("ChatMessage", backref="evaluations")

    def __repr__(self) -> str:
        return f"<MessageEvaluation(id={self.id}, chat_message_id={self.chat_message_id}, ndcg_score={self.ndcg_score})>"
