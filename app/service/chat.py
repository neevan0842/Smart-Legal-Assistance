from typing import List
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models.chat import ChatMessage
from sqlalchemy.orm import selectinload
from app.db.models.document import MessageDocument
from app.schema.chats import ChatMessageResponse, DocumentResponse
from sqlalchemy.future import select


async def get_chat_messages_with_score(
    chat_id: UUID, db: AsyncSession
) -> List[ChatMessageResponse]:
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == chat_id)
        .order_by(ChatMessage.created_at.asc())
        .options(
            selectinload(ChatMessage.message_documents).selectinload(
                MessageDocument.document
            ),
            selectinload(ChatMessage.evaluations),
        )
    )
    result = await db.execute(stmt)
    messages = result.scalars().unique().all()
    response: List[ChatMessageResponse] = []

    for msg in messages:
        ndgc_score = msg.evaluations[0].ndcg_score if msg.evaluations else None
        documents = [
            DocumentResponse.model_validate(md.document)
            for md in msg.message_documents
            if md.document is not None
        ]
        response.append(
            ChatMessageResponse(
                id=msg.id,
                session_id=msg.session_id,
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at,
                ndcg_score=ndgc_score,
                documents=documents,
            )
        )

    return response
