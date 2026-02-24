from uuid import UUID
from sqlalchemy.future import select
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from app.core.dependencies import get_db, get_pinecone_service
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models.chat import ChatSession
from app.db.models.document import Document
from app.db.models.user import User
from app.schema.documents import DocumentUploadResponse
from app.service.documents import (
    delete_document_file_from_storage,
    delete_document_from_pinecone,
    save_uploaded_file,
    upsert_document_to_pinecone,
)
from app.service.users import get_current_active_user


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    chat_session_id: UUID = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    pc_svc=Depends(get_pinecone_service),
):
    """Endpoint to upload a document and associate it with a chat session."""
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are allowed.",
        )

    # verify chat session exists and belongs to the user
    stmt = select(ChatSession).where(
        ChatSession.id == chat_session_id,
        ChatSession.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    chat_session = result.scalar_one_or_none()
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found."
        )

    # Save the uploaded file and create a document record in the database
    document = await save_uploaded_file(file, current_user, str(chat_session_id), db)
    # Upsert the document content to Pinecone vector database under the chat session namespace
    await upsert_document_to_pinecone(
        document=document,
        chat_session_id=str(chat_session_id),
        pc_svc=pc_svc,
    )

    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    pc_svc=Depends(get_pinecone_service),
):
    """Endpoint to delete a document by its ID. Only the owner of the document can delete it."""
    # verify document exists and belongs to the user
    stmt = select(Document).where(
        Document.id == document_id, Document.user_id == current_user.id
    )
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found."
        )

    # delete the document file from storage
    delete_document_file_from_storage(document)

    # get the associated chat session ID for the document
    stmt = select(Document.chat_session_id).where(Document.id == document_id)
    result = await db.execute(stmt)
    chat_session_id = result.scalar_one_or_none()
    if chat_session_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Associated chat session not found for the document.",
        )

    # delete the document records from Pinecone vector database
    await delete_document_from_pinecone(
        document_id=str(document.id),
        chat_session_id=str(chat_session_id),
        pc_svc=pc_svc,
    )

    await db.delete(document)
    await db.commit()
