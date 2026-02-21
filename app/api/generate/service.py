from typing import AsyncGenerator
from app.utils.groq import GroqService
from app.utils.pinecone import PineconeService
from app.utils.utils import create_system_message


async def generate_answer(
    query: str,
    pc_svc: PineconeService,
    groq_svc: GroqService,
    top_k: int = 20,
    top_n: int = 10,
) -> str:
    """Generate an answer for a legal query using RAG."""

    context_from_pinecone = await pc_svc.query_and_rerank(
        query=query, top_k=top_k, top_n=top_n
    )
    system_message = create_system_message(context_from_pinecone)

    # Get complete response from LLM
    return await groq_svc.generate_answer(query=query, system_prompt=system_message)


async def generate_answer_stream(
    query: str,
    pc_svc: PineconeService,
    groq_svc: GroqService,
    top_k: int = 20,
    top_n: int = 10,
) -> AsyncGenerator[str, None]:
    """Generate an answer for a legal query using RAG, streaming the response."""
    context_from_pinecone = await pc_svc.query_and_rerank(
        query=query, top_k=top_k, top_n=top_n
    )
    system_message = create_system_message(context_from_pinecone)

    # Stream response from LLM
    async for chunk in groq_svc.generate_answer_stream(
        query=query, system_prompt=system_message
    ):
        yield chunk
