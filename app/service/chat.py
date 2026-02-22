from typing import List
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models.chat import ChatMessage
from app.db.models.evaluation import MessageEvaluation
from app.schema.chats import ChatMessageResponse
from sqlalchemy.future import select


async def get_chat_messages_with_score(
    chat_id: UUID, db: AsyncSession
) -> List[ChatMessageResponse]:
    stmt = (
        select(ChatMessage, MessageEvaluation.ndcg_score)
        .outerjoin(
            MessageEvaluation, ChatMessage.id == MessageEvaluation.chat_message_id
        )
        .where(ChatMessage.session_id == chat_id)
        .order_by(ChatMessage.created_at.asc())
    )
    result = await db.execute(stmt)
    messages_with_scores = result.all()

    return [
        ChatMessageResponse(
            id=chat_message.id,
            session_id=chat_message.session_id,
            role=chat_message.role,
            content=chat_message.content,
            created_at=chat_message.created_at,
            ndcg_score=ndcg_score,
        )
        for chat_message, ndcg_score in messages_with_scores
    ]
