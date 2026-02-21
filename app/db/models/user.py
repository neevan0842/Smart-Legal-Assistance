import uuid
from sqlalchemy import String, TIMESTAMP, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from app.db.base import Base


class User(Base):
    __tablename__ = "users"
    # syntax : attribute: Mapped[type] = mapped_column(SQLAlchemyType, options…)
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )

    documents = relationship("Document", back_populates="user", cascade="all, delete")
    chat_sessions = relationship(
        "ChatSession", back_populates="user", cascade="all, delete"
    )
    generated_documents = relationship(
        "GeneratedDocument", back_populates="user", cascade="all, delete"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"
