import asyncio
import json
from app.core.logger import logger
from typing import List
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.constants import USER_UPLOADS_DIR, DocumentSourceType
from app.db.models.chat import ChatSession
from app.db.models.document import Document
from app.db.models.user import User
from app.schema.documents import DocumentAnalysis
from app.utils.groq import GroqService
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


async def document_analysis_map_reduce(
    chunks: List[str], groq_svc: GroqService
) -> DocumentAnalysis:
    semaphore = asyncio.Semaphore(3)
    total_chunks, chunks_done = len(chunks), 0

    async def map_chunk(chunk: str, semaphore: asyncio.Semaphore):
        async with semaphore:
            result = await groq_svc.document_analysis_llm_response(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a legal document analysis assistant. "
                            "Your task is to analyze the provided legal text and return a JSON object that strictly matches the following schema: "
                            "{\n"
                            "  'summary': str,\n"
                            "  'risks': List[{'clause_reference': Optional[str], 'risk_level': 'Low'|'Medium'|'High', 'issue': str, 'recommendation': str}],\n"
                            "  'suggestions': List[str],\n"
                            "  'missing_clauses': List[str]\n"
                            "}\n"
                            "You MUST always include all four keys: 'summary', 'risks', 'suggestions', and 'missing_clauses', even if they are empty. "
                            "If a field has no value, return an empty list or empty string as appropriate. "
                            "Do not include any extra fields or text. "
                            "Ensure 'risks' is a list of objects with the specified keys and types. "
                            "Return ONLY valid JSON. Do not explain your answer."
                        ),
                    },
                    {"role": "user", "content": f"Analyze this legal text:\n{chunk}"},
                ],
                schema_model=DocumentAnalysis,
            )
            nonlocal chunks_done
            chunks_done += 1
            logger.debug(f"chunks done : {chunks_done} / {total_chunks}")
            return result

    async def reduce_results(mapped_results: List[DocumentAnalysis]):
        combined = DocumentAnalysis(
            summary="\n\n".join(result.summary for result in mapped_results),
            risks=[risk for result in mapped_results for risk in result.risks],
            suggestions=[
                suggestion
                for result in mapped_results
                for suggestion in result.suggestions
            ],
            missing_clauses=[
                clause for result in mapped_results for clause in result.missing_clauses
            ],
        )
        combined = json.dumps(combined.model_dump())
        logger.debug(f"combined mapped results for reduce step: {combined}")
        result = await groq_svc.document_analysis_llm_response(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal document synthesis assistant. "
                        "Your task is to combine the analysis results from multiple chunks into a single, minimal JSON object that strictly matches the following schema: "
                        "{\n"
                        "  'summary': str,\n"
                        "  'risks': List[{'clause_reference': Optional[str], 'risk_level': 'Low'|'Medium'|'High', 'issue': str, 'recommendation': str}],\n"
                        "  'suggestions': List[str],\n"
                        "  'missing_clauses': List[str]\n"
                        "}\n"
                        "Only include the most essential and necessary information for each field. "
                        "Keep the summary, suggestions, and risks as concise and minimal as possible. "
                        "Avoid verbosity, repetition, or unnecessary details. "
                        "If a field is not strictly necessary, leave it empty. "
                        "Do not include any extra fields or text. "
                        "Return ONLY valid JSON. Do not explain your answer."
                    ),
                },
                {"role": "user", "content": f"Synthesize this analysis:\n{combined}"},
            ],
            schema_model=DocumentAnalysis,
        )
        logger.debug("Completed map-reduce analysis of document.")
        return result

    mapped_results = await asyncio.gather(
        *(map_chunk(chunk, semaphore) for chunk in chunks)
    )
    final_analysis = await reduce_results(mapped_results)
    return final_analysis
