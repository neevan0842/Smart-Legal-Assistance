from fastapi import APIRouter, HTTPException, Request
from .schema import QueryRequest, QueryResponse
from .service import generate_answer


router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("/", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest, request: Request) -> QueryResponse:
    """
    Query the legal assistant with a question and get an AI-generated answer
    based on BNS and BNSS legal documents.
    """
    try:
        answer = await generate_answer(
            request=request, query=body.query, top_k=body.top_k, top_n=body.top_n
        )
        return QueryResponse(answer=answer, query=body.query)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating answer: {str(e)}"
        )
