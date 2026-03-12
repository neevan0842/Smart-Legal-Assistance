from datetime import datetime
import io
import tempfile
from uuid import UUID
from app.core.logger import logger
from fastapi.responses import FileResponse
from sqlalchemy.future import select
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from app.core.constants import GENERATED_DOCS_DIR, USER_UPLOADS_DIR, DocumentSourceType
from app.core.dependencies import get_db, get_groq_service, get_pinecone_service
from app.db.models.document import Document
from sqlalchemy.ext.asyncio import AsyncSession
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
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


@router.post("/generate")
async def generate_new_contract(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    contract_type: str = Form(...),
    buyer_name: str = Form(...),
    buyer_address: str = Form(...),
    buyer_aadhar: str = Form(...),
    seller_name: str = Form(...),
    seller_address: str = Form(...),
    seller_aadhar: str = Form(...),
    amount: str = Form(...),
    property_details: str = Form(...),
):
    """Endpoint to generate a new contract based on user input and predefined templates."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # Stamp Header
    header_style = ParagraphStyle(
        "header",
        parent=styles["Normal"],
        fontSize=16,
        alignment=1,
        textColor=colors.darkred,
    )

    elements.append(Paragraph("🦁 GOVERNMENT OF INDIA", header_style))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("NON-JUDICIAL STAMP PAPER", header_style))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("₹ 100", header_style))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(
        Paragraph(f"Date: {datetime.now().strftime('%d-%m-%Y')}", styles["Normal"])
    )
    elements.append(Spacer(1, 0.5 * inch))

    # Agreement Title
    elements.append(
        Paragraph(f"<b>{contract_type.upper()} AGREEMENT</b>", styles["Heading2"])
    )
    elements.append(Spacer(1, 0.3 * inch))

    # Agreement Text
    if contract_type.lower() == "buy":
        agreement_text = f"""
        This Sale Agreement is made between <b>{seller_name}</b>, residing at {seller_address},
        Aadhaar No: {seller_aadhar} (Seller),

        AND

        <b>{buyer_name}</b>, residing at {buyer_address},
        Aadhaar No: {buyer_aadhar} (Buyer).

        The Seller agrees to sell the property described below:

        {property_details}

        for a total consideration of ₹{amount}.
        """
    else:
        agreement_text = f"""
        This Rental Agreement is made between <b>{seller_name}</b>, residing at {seller_address},
        Aadhaar No: {seller_aadhar} (Landlord),

        AND

        <b>{buyer_name}</b>, residing at {buyer_address},
        Aadhaar No: {buyer_aadhar} (Tenant).

        The Landlord agrees to rent the property described below:

        {property_details}

        for a monthly rent of ₹{amount}.
        """

    elements.append(Paragraph(agreement_text, styles["Normal"]))
    elements.append(Spacer(1, 1.5 * inch))

    # Signature Section
    signature_table = Table(
        [
            ["Signature of Seller/Landlord", "", "Signature of Buyer/Tenant"],
            ["__________________", "", "__________________"],
        ],
        colWidths=[2.5 * inch, 1 * inch, 2.5 * inch],
    )

    signature_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LINEABOVE", (0, 1), (0, 1), 1, colors.black),
                ("LINEABOVE", (2, 1), (2, 1), 1, colors.black),
            ]
        )
    )

    elements.append(signature_table)
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("Witness 1: ____________________", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("Witness 2: ____________________", styles["Normal"]))

    doc.build(elements)

    # Persist the generated document to storage and database
    filename = f"{contract_type.lower()}_agreement.pdf"
    document = Document(
        user_id=current_user.id,
        source_type=DocumentSourceType.GENERATED,
        filename=filename,
        storage_path=GENERATED_DOCS_DIR.as_posix(),
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    file_location = GENERATED_DOCS_DIR / f"{document.id}_{filename}"
    with open(file_location, "wb") as f:
        f.write(buffer.getvalue())

    logger.info(
        f"Generated new contract for user {current_user.id} with document ID {document.id} at {file_location}"
    )

    return FileResponse(
        path=file_location,
        media_type="application/pdf",
        filename=filename,
    )


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
        media_type="application/pdf",
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
