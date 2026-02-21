from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.api.generate.schema import QueryRequest, QueryResponse
from app.api.generate.service import generate_answer, generate_answer_stream
from app.core.dependencies import get_groq_service, get_pinecone_service


router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("/", response_model=QueryResponse)
async def query_endpoint(
    body: QueryRequest,
    pc_svc=Depends(get_pinecone_service),
    groq_svc=Depends(get_groq_service),
):
    """
    Query the legal assistant with a question and get an AI-generated answer
    based on BNS and BNSS legal documents.

    Set `stream=true` to receive streaming response chunks,
    or `stream=false` (default) for complete response.
    """
    try:
        if body.stream:
            return StreamingResponse(
                generate_answer_stream(
                    query=body.query,
                    pc_svc=pc_svc,
                    groq_svc=groq_svc,
                    top_k=body.top_k,
                    top_n=body.top_n,
                ),
                media_type="text/event-stream",
            )
        else:
            answer = await generate_answer(
                query=body.query,
                pc_svc=pc_svc,
                groq_svc=groq_svc,
                top_k=body.top_k,
                top_n=body.top_n,
            )
            return QueryResponse(answer=answer, query=body.query)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating answer: {str(e)}"
        )
