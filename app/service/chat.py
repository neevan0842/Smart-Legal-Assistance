from typing import Dict, List
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.constants import ChatRole
from app.db.models.chat import ChatMessage
from sqlalchemy.orm import selectinload
from app.db.models.document import MessageDocument
from app.schema.chats import (
    ChatMessageResponse,
    DocumentResponse,
)
from sqlalchemy.future import select
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from app.utils.utils import base_messages_to_groq


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


async def construct_prompt(
    query: str, context: str, chat_id: UUID, db: AsyncSession
) -> List[Dict]:

    # Get previous chat history
    chat_history = await get_previous_chat_history(chat_id=chat_id, db=db)

    # RAG Prompt — system + history + retrieve + question
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are an expert legal assistant. Answer using the provided facts and chat history."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(
                "Relevant excerpts from documents:\n"
                "{retrieved_context}\n\n"
                "User question:\n"
                "{query}"
            ),
        ]
    )

    prompt = prompt_template.format_prompt(
        query=query, retrieved_context=context, chat_history=chat_history
    )
    prompt_messages = base_messages_to_groq(prompt.to_messages())

    return prompt_messages


async def get_previous_chat_history(
    chat_id: UUID, db: AsyncSession
) -> List[ChatMessage]:
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == chat_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(5)
    )
    result = await db.execute(stmt)
    messages = result.scalars().all()
    chat_history = []
    for msg in reversed(messages):
        if msg.role == ChatRole.USER:
            chat_history.append(HumanMessage(content=msg.content))
        else:
            chat_history.append(AIMessage(content=msg.content))
    return chat_history
