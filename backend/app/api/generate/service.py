import asyncio
from fastapi import Request
from ...core.config import settings


def merge_chunks(h1, h2):
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


async def generate_answer(
    request: Request, query: str, top_k: int = 20, top_n: int = 10
) -> str:
    """Generate an answer for a legal query using RAG."""
    # Get app state objects
    async_dense_index = request.app.state.async_dense_index
    async_sparse_index = request.app.state.async_sparse_index
    pc_async = request.app.state.pc_async
    groq_client = request.app.state.groq_client

    # Run both searches concurrently
    dense_response, sparse_response = await asyncio.gather(
        async_dense_index.search_records(
            namespace="bns_and_bnss",
            query={"top_k": top_k, "inputs": {"text": query}},
        ),
        async_sparse_index.search_records(
            namespace="bns_and_bnss",
            query={"top_k": top_k, "inputs": {"text": query}},
        ),
    )

    # Merge results
    merged_results = merge_chunks(sparse_response, dense_response)

    # Rerank results
    reranked_result = await pc_async.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=merged_results,
        rank_fields=["content"],
        top_n=top_n,
        return_documents=True,
        parameters={"truncate": "END"},
    )

    # Build context from reranked results
    combined_context = "\n\n".join(
        f"{hit['document']['_id'].split('_')[0]} - Section {int(hit['document']['metadata']['section_number'])} - {hit['document']['content']}"
        for hit in reranked_result.data
    )

    # Create system prompt
    system_message = f"Answer any use questions based solely on the context below:\n\n<context>\n{combined_context.strip()}\n</context>"

    # Get response from LLM
    chat_completion = await groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        model=settings.LLM_MODEL_NAME,
    )

    return chat_completion.choices[0].message.content
