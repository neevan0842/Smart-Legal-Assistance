import asyncio
from typing import List
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.constants import USER_UPLOADS_DIR, DocumentSourceType
from app.db.models.chat import ChatSession
from app.db.models.document import Document
from app.db.models.user import User
from app.utils.pinecone import PineconeService
from app.utils.utils import PDFLoader, split_text_to_chunks
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload


async def handle_multiple_file_uploads(
    files: List[UploadFile],
    current_user: User,
    chat_session_id: str,
    db: AsyncSession,
    pc_svc: PineconeService,
) -> List[Document]:
    documents = []
    for file in files:
        document = await save_uploaded_file(file, current_user, chat_session_id, db)
        documents.append(document)

    tasks = [
        upsert_document_to_pinecone(document, chat_session_id, pc_svc)
        for document in documents
    ]
    await asyncio.gather(*tasks)
    return documents


async def save_uploaded_file(
    file: UploadFile, current_user: User, chat_session_id: str, db: AsyncSession
) -> Document:
    document = Document(
        user_id=current_user.id,
        chat_session_id=chat_session_id,
        source_type=DocumentSourceType.UPLOAD,
        filename=file.filename,
        storage_path=USER_UPLOADS_DIR.as_posix(),
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    # Save the file to the server
    file_location = USER_UPLOADS_DIR / f"{document.id}_{file.filename}"
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)

    return document


async def upsert_document_to_pinecone(
    document: Document,
    chat_session_id: str,
    pc_svc: PineconeService,
):
    document_id = str(document.id)
    filepath = (USER_UPLOADS_DIR / f"{document_id}_{document.filename}").as_posix()
    text = PDFLoader(filepath).extract_text()
    chunks = split_text_to_chunks(text)
    records = [
        {
            "id": f"{document_id}_{i}",
            "content": content,
            "document_id": document_id,
        }
        for i, content in enumerate(chunks)
    ]
    await pc_svc.upsert_records(records, namespace=chat_session_id)


def delete_document_file_from_storage(document: Document):
    file_location = USER_UPLOADS_DIR / f"{document.id}_{document.filename}"
    if file_location.exists():
        file_location.unlink()


async def delete_all_documents_from_storage_by_chat_session_id(
    chat_session_id: str, db: AsyncSession
):
    stmt = (
        select(ChatSession)
        .where(ChatSession.id == chat_session_id)
        .options(selectinload(ChatSession.documents))
    )
    result = await db.execute(stmt)
    chat_session = result.scalar_one_or_none()
    if chat_session:
        for document in chat_session.documents:
            delete_document_file_from_storage(document)


async def delete_document_from_pinecone(
    document_id: str, chat_session_id: str, pc_svc: PineconeService
) -> bool:
    result = await pc_svc.delete_document_records_by_metadata(
        document_id=document_id, namespace=chat_session_id
    )
    return result
