from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The legal question to ask")
    top_k: int = Field(
        20, ge=1, le=100, description="Number of results to retrieve from each index"
    )
    top_n: int = Field(
        10, ge=1, le=50, description="Number of results to keep after reranking"
    )
    stream: bool = Field(
        False, description="Whether to stream the response or return it complete"
    )


class QueryResponse(BaseModel):
    answer: str = Field(
        ..., description="The generated answer based on the legal documents"
    )
    query: str = Field(..., description="The original query")
