from uuid import UUID
from fastapi.responses import FileResponse
from sqlalchemy.future import select
from fastapi import APIRouter, Depends, HTTPException, status
from app.core.constants import USER_UPLOADS_DIR
from app.core.dependencies import get_db, get_pinecone_service
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models.document import Document
from app.db.models.user import User
from app.service.documents import (
    delete_document_file_from_storage,
    delete_document_from_pinecone,
)
from app.service.users import get_current_active_user


router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/{document_id}")
async def download_document(
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Endpoint to download a document by its ID. Only the owner of the document can download it."""
    stmt = select(Document).where(
        Document.id == document_id, Document.user_id == current_user.id
    )
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found."
        )

    file_location = USER_UPLOADS_DIR / f"{document.id}_{document.filename}"
    if not file_location.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found on server.",
        )

    return FileResponse(
        path=file_location,
        media_type="application/octet-stream",
        filename=document.filename,
    )


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
