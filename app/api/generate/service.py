import asyncio
from typing import AsyncGenerator, Tuple
from fastapi import Request
from ...core.config import settings

NAMESPACE = "legal_law_documents"


def _merge_chunks(h1, h2):
    """Get the unique hits from two search results and return them as single array."""
    # Deduplicate by _id
    deduped_hits = {
        hit["_id"]: hit for hit in h1["result"]["hits"] + h2["result"]["hits"]
    }.values()
    # Sort by _score descending
    sorted_hits = sorted(deduped_hits, key=lambda x: x["_score"], reverse=True)
    # Transform to format for reranking
    result = [
        {
            "_id": hit["_id"],
            "content": hit["fields"]["content"],
            "metadata": {
                "chunk": hit["fields"]["chunk"],
                "is_chunked": hit["fields"]["is_chunked"],
                "section_number": hit["fields"]["section_number"],
            },
        }
        for hit in sorted_hits
    ]
    return result


def _build_context_from_results(reranked_result) -> str:
    """Build context string from reranked results."""
    return "\n\n".join(
        f"{hit['document']['_id'].split('_')[0]} - Section {int(hit['document']['metadata']['section_number'])} - {hit['document']['content']}"
        for hit in reranked_result.data
    )


def _create_system_message(context: str) -> str:
    """Create system prompt with context."""
    return f"Answer any use questions based solely on the context below:\n\n<context>\n{context.strip()}\n</context>"


async def _retrieve_and_rerank(
    request: Request, query: str, top_k: int = 20, top_n: int = 10
) -> Tuple[str, any]:
    """
    Retrieve relevant documents using RAG and rerank them.
    Returns the context string and app state objects for LLM calls.
    """
    # Get app state objects
    async_dense_index = request.app.state.async_dense_index
    async_sparse_index = request.app.state.async_sparse_index
    pc_async = request.app.state.pc_async
    groq_client = request.app.state.groq_client

    # Run both searches concurrently
    dense_response, sparse_response = await asyncio.gather(
        async_dense_index.search_records(
            namespace=NAMESPACE,
            query={"top_k": top_k, "inputs": {"text": query}},
        ),
        async_sparse_index.search_records(
            namespace=NAMESPACE,
            query={"top_k": top_k, "inputs": {"text": query}},
        ),
    )

    # Merge and rerank results
    merged_results = _merge_chunks(sparse_response, dense_response)
    reranked_result = await pc_async.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=merged_results,
        rank_fields=["content"],
        top_n=top_n,
        return_documents=True,
        parameters={"truncate": "END"},
    )

    # Build context and system message
    context = _build_context_from_results(reranked_result)
    system_message = _create_system_message(context)

    return system_message, groq_client


async def generate_answer(
    request: Request, query: str, top_k: int = 20, top_n: int = 10
) -> str:
    """Generate an answer for a legal query using RAG."""
    system_message, groq_client = await _retrieve_and_rerank(
        request, query, top_k, top_n
    )

    # Get complete response from LLM
    chat_completion = await groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ],
        model=settings.LLM_MODEL_NAME,
    )

    return chat_completion.choices[0].message.content


async def generate_answer_stream(
    request: Request, query: str, top_k: int = 20, top_n: int = 10
) -> AsyncGenerator[str, None]:
    """Generate an answer for a legal query using RAG with streaming."""
    system_message, groq_client = await _retrieve_and_rerank(
        request, query, top_k, top_n
    )

    # Get streaming response from LLM
    stream = await groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ],
        model=settings.LLM_MODEL_NAME,
        stream=True,
    )

    # Yield chunks as they arrive in SSE format
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            # Format as Server-Sent Events
            yield f"data: {content}\n\n"

    # Send completion marker
    yield "data: [DONE]\n\n"
