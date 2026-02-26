import tempfile
from uuid import UUID
from fastapi.responses import FileResponse
from sqlalchemy.future import select
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from app.core.constants import USER_UPLOADS_DIR
from app.core.dependencies import get_db, get_groq_service, get_pinecone_service
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models.document import Document
from app.db.models.user import User
from app.schema.documents import DocumentAnalysis
from app.service.documents import (
    delete_document_file_from_storage,
    delete_document_from_pinecone,
    document_analysis_map_reduce,
)
from app.service.users import get_current_active_user
from app.utils.groq import GroqService
from app.utils.utils import PDFLoader, split_text_to_chunks


router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/analyze", response_model=DocumentAnalysis)
async def analyze_document(
    files: list[UploadFile] = File(...),
    _: User = Depends(get_current_active_user),
    groq_svc: GroqService = Depends(get_groq_service),
):
    """Endpoint to analyze an uploaded document. Only one file is allowed."""

    if not files or len(files) != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Exactly one file must be uploaded for analysis.",
        )

    file = files[0]
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported for analysis.",
        )

    with tempfile.TemporaryDirectory() as temp_dir:

        # Save the uploaded file to a temporary location
        temp_path = f"{temp_dir}/{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extract text from the PDF
        extracted_text = PDFLoader(temp_path).extract_text()

        # Split the extracted text into chunks
        chunks = split_text_to_chunks(
            extracted_text, chunk_size=8000, chunk_overlap=500
        )

    # Perform analysis on the chunks using the map-reduce approach
    results = await document_analysis_map_reduce(chunks, groq_svc=groq_svc)
    return results


@router.get("/generate")
async def generate_new_contract(
    _: User = Depends(get_current_active_user),
):
    """Endpoint to generate a new contract based on user input and predefined templates."""
    # TODO: Implement an endpoint that generates a new contract based on user input and predefined templates.
    pass


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
