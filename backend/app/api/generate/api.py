from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from .schema import QueryRequest, QueryResponse
from .service import generate_answer, generate_answer_stream


router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("/", response_model=None)
async def query_endpoint(body: QueryRequest, request: Request):
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
                    request=request,
                    query=body.query,
                    top_k=body.top_k,
                    top_n=body.top_n,
                ),
                media_type="text/event-stream",
            )
        else:
            answer = await generate_answer(
                request=request,
                query=body.query,
                top_k=body.top_k,
                top_n=body.top_n,
            )
            return QueryResponse(answer=answer, query=body.query)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating answer: {str(e)}"
        )
