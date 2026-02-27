import asyncio
from typing import List
from app.core.logger import logger
from uuid import UUID
from sqlalchemy.future import select
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.constants import NAMESPACE, ChatRole
from app.core.dependencies import get_db, get_groq_service, get_pinecone_service
from app.db.models.chat import ChatMessage, ChatSession
from app.db.models.document import MessageDocument
from app.db.models.evaluation import MessageEvaluation
from app.db.models.user import User
from app.schema.chats import (
    ChatMessageAIResponse,
    ChatMessageResponse,
    ChatSessionResponse,
    UpdateChatSessionTitleRequest,
)
from app.service.chat import construct_prompt, get_chat_messages_with_score
from app.service.documents import (
    delete_all_documents_from_storage_by_chat_session_id,
    handle_multiple_file_uploads,
)
from app.service.users import get_current_active_user
from app.utils.groq import GroqService
from app.utils.pinecone import PineconeService
from app.utils.utils import get_ndcg_score_at_k


router = APIRouter(prefix="/chats", tags=["chats"])


@router.get("/", response_model=List[ChatSessionResponse])
async def get_all_user_chats_sessions(
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Endpoint to retrieve all chat sessions for the authenticated user, ordered by creation date (most recent first)."""
    stmt = (
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.created_at.desc())
    )
    result = await db.execute(stmt)
    chat_sessions = result.scalars().all()
    return chat_sessions


@router.post(
    "/", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED
)
async def create_new_chat_session(
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    pc_svc: PineconeService = Depends(get_pinecone_service),
):
    """Endpoint to create a new chat session for the authenticated user."""
    new_chat = ChatSession(user_id=user.id)
    db.add(new_chat)
    await db.commit()
    await db.refresh(new_chat)

    # create a namespace in Pinecone
    await pc_svc.create_pinecone_namespace(namespace=str(new_chat.id))

    new_message = ChatMessage(
        session_id=new_chat.id,
        role=ChatRole.ASSISTANT,
        content="Hello! How can I assist you today?",
    )
    db.add(new_message)
    await db.commit()
    return new_chat


@router.patch("/{chat_id}", response_model=ChatSessionResponse)
async def update_chat_session_title(
    chat_id: UUID,
    payload: UpdateChatSessionTitleRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Endpoint to update the title of a chat session for the authenticated user."""
    stmt = select(ChatSession).where(
        ChatSession.id == chat_id, ChatSession.user_id == user.id
    )
    result = await db.execute(stmt)
    chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found"
        )
    chat_session.title = payload.title
    await db.commit()
    await db.refresh(chat_session)
    return chat_session


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    chat_id: UUID,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    pc_svc: PineconeService = Depends(get_pinecone_service),
):
    """Endpoint to delete a chat session for the authenticated user."""
    stmt = select(ChatSession).where(
        ChatSession.id == chat_id, ChatSession.user_id == user.id
    )
    result = await db.execute(stmt)
    chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found"
        )

    # delete all associated namespace in Pinecone vector database and storage
    await pc_svc.delete_pinecone_namespace(namespace=str(chat_session.id))
    await delete_all_documents_from_storage_by_chat_session_id(
        chat_session_id=str(chat_session.id), db=db
    )

    await db.delete(chat_session)
    await db.commit()
    return None


@router.get("/{chat_id}/messages", response_model=List[ChatMessageResponse])
async def get_chat_messages_by_chat_session_id(
    chat_id: UUID,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Endpoint to retrieve the list of messages for a specific chat session of the authenticated user."""
    # Verify that the chat session exists and belongs to the user
    stmt = select(ChatSession).where(
        ChatSession.id == chat_id, ChatSession.user_id == user.id
    )
    result = await db.execute(stmt)
    chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found"
        )

    # Retrieve messages for the chat session
    messages = await get_chat_messages_with_score(chat_id=chat_id, db=db)
    return messages


@router.post("/{chat_id}/messages", response_model=ChatMessageAIResponse)
async def add_message_to_chat_session_and_generate_llm_response(
    chat_id: UUID,
    query: str = Form(...),
    files: List[UploadFile] = File([]),
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    pc_svc: PineconeService = Depends(get_pinecone_service),
    groq_svc: GroqService = Depends(get_groq_service),
):
    """Endpoint to add a new message to a specific chat session and generate an LLM response."""
    # verify files are pdfs
    files = [file for file in files if file.size != 0]
    for doc in files:
        if doc.content_type != "application/pdf":
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: {doc.filename}. Only PDF files are allowed.",
            )

    # Verify that the chat session exists and belongs to the user
    stmt = select(ChatSession).where(
        ChatSession.id == chat_id, ChatSession.user_id == user.id
    )
    result = await db.execute(stmt)
    chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found"
        )

    # Handle uploaded files
    documents = await handle_multiple_file_uploads(
        files, user, str(chat_id), db, pc_svc
    )

    # Add user's message to the chat session
    user_message = ChatMessage(session_id=chat_id, role=ChatRole.USER, content=query)
    db.add(user_message)
    await db.commit()
    await db.refresh(user_message)

    # Associate uploaded documents with the user's message
    message_documents = []
    for document in documents:
        message_document = MessageDocument(
            message_id=user_message.id, document_id=document.id
        )
        message_documents.append(message_document)
    db.add_all(message_documents)
    await db.commit()

    tasks = [
        pc_svc.query_index_and_merge(query=query, top_k=15, namespace=str(chat_id)),
        pc_svc.query_index_and_merge(query=query, top_k=15, namespace=NAMESPACE),
    ]
    user_session_results, global_results = await asyncio.gather(*tasks)
    logger.debug(f"User session results: {user_session_results}")
    logger.debug(f"Global results: {global_results}")

    context = await pc_svc.rerank_merged_records_and_get_context(
        query=query, merged_results=user_session_results + global_results
    )

    # get relevance score for the retrieved documents
    relavance_score = await groq_svc.evaluate_rag_results_to_get_relevance_scores(
        query=query, retrieved_documents=context
    )
    ndcg_score = get_ndcg_score_at_k(
        relevance_scores=relavance_score, k=len(relavance_score)
    )
    logger.debug(f"Relevance scores: {relavance_score}")

    combined_context = "\n\n".join(context)
    logger.debug(f"Combined context for Groq: {combined_context}")

    prompt_messages = await construct_prompt(
        query=query, context=combined_context, chat_id=chat_id, db=db
    )

    logger.debug(f"Constructed prompt for Groq: {prompt_messages}")

    ai_message_content = await groq_svc.generate_answer(prompt_messages=prompt_messages)

    ai_message = ChatMessage(
        session_id=chat_id, role=ChatRole.ASSISTANT, content=ai_message_content
    )
    db.add(ai_message)
    await db.commit()
    await db.refresh(ai_message)

    # store ndcg score in the database
    evaluation = MessageEvaluation(
        message_id=ai_message.id,
        ndcg_score=ndcg_score,
    )
    db.add(evaluation)
    await db.commit()

    return ChatMessageAIResponse(
        id=ai_message.id,
        session_id=ai_message.session_id,
        role=ai_message.role,
        content=ai_message.content,
        created_at=ai_message.created_at,
        context=context,
        ndcg_score=ndcg_score,
        documents=documents,
    )
