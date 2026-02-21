import asyncio
import fitz
import re
import json
from typing import List, Dict
from groq import AsyncGroq
from pinecone import PineconeAsyncio
from app.core.config import settings


PINECONE_API_KEY = settings.PINECONE_API_KEY
PINECONE_DENSE_INDEX_NAME = settings.PINECONE_DENSE_INDEX_NAME
PINECONE_SPARSE_INDEX_NAME = settings.PINECONE_SPARSE_INDEX_NAME
PINECONE_DENSE_INDEX_MODEL = settings.PINECONE_DENSE_INDEX_MODEL
PINECONE_SPARSE_INDEX_MODEL = settings.PINECONE_SPARSE_INDEX_MODEL
PINECONE_DENSE_HOST = settings.PINECONE_DENSE_HOST
PINECONE_SPARSE_HOST = settings.PINECONE_SPARSE_HOST
LLM_MODEL_NAME = settings.LLM_MODEL_NAME
GROQ_API_KEY = settings.GROQ_API_KEY


def extract_sections_general(pdf_path: str) -> List[Dict]:
    """
    General extractor for legal/numbered sections from a PDF.
    Works for both BNS and BNSS and similar structured documents.
    """

    # --- 1) Read full PDF text by lines ---
    doc = fitz.open(pdf_path)
    lines = []
    for page in doc:
        text = page.get_text("text")
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("_", "")
        lines.extend(text.split("\n"))
    doc.close()

    # --- 2) Define regex patterns for section headings ---

    # Pattern A: number and text on same line
    # e.g., "123. Some text starts here" OR "123 ) Some text"
    header_with_text = re.compile(r"^\s*(\d{1,4})\s*(?:\.\s*|\)\s*|\-\s*)(\S.+)$")

    # Pattern B: number on its own line (then text follows on next lines)
    # e.g., "123."
    header_only = re.compile(r"^\s*(\d{1,4})\s*(?:\.\s*|\)\s*|\-\s*)\s*$")

    sections = {}
    current_section = None

    # --- 3) Iterate through all lines ---
    for line in lines:

        # Try pattern A: number + text on same line
        m = header_with_text.match(line)
        if m:
            num = int(m.group(1))
            text = m.group(2).strip()

            current_section = num
            # Start new section content
            sections[current_section] = text
            continue

        # Try pattern B: number only on line
        m2 = header_only.match(line)
        if m2:
            num = int(m2.group(1))
            current_section = num
            sections[current_section] = ""
            continue

        # If currently inside a section, append text
        if current_section is not None:
            # avoid adding blank junk lines
            stripped = line.strip()
            if stripped:
                # accumulate with a space
                sections[current_section] += (
                    " " + stripped if sections[current_section] else stripped
                )

    # --- 4) Convert into sorted list of dicts ---
    result = []
    for num in sorted(sections.keys()):
        # Normalize whitespace
        content = re.sub(r"\s+", " ", sections[num].strip())
        result.append(
            {
                "id": f"{'bnss' if 'bnss' in pdf_path.lower() else 'bns'}_{num}",
                "section_number": num,
                "content": content,
            }
        )

    return result


def save_as_json(data, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def chunk_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def chunk_long_sections(
    sections: List[Dict], max_content_bytes: int = 35000
) -> List[Dict]:
    """
    Split sections with content exceeding max_content_bytes into smaller chunks.
    Keeps metadata under Pinecone's 40KB limit while preserving section context.
    All sections have uniform structure with chunk and is_chunked fields.
    """
    chunked_sections = []

    for section in sections:
        content = section["content"]
        content_bytes = len(content.encode("utf-8"))

        if content_bytes <= max_content_bytes:
            # Section is small enough, keep as is with uniform structure
            chunked_sections.append(
                {
                    "id": section["id"],
                    "section_number": section["section_number"],
                    "content": content,
                    "chunk": 1,
                    "is_chunked": False,
                }
            )
        else:
            # Need to chunk this section
            # Split by sentences to avoid breaking mid-sentence
            sentences = re.split(r"(?<=[.!?])\s+", content)

            chunk_text = ""
            chunk_num = 1

            for sentence in sentences:
                test_text = chunk_text + " " + sentence if chunk_text else sentence

                if len(test_text.encode("utf-8")) > max_content_bytes and chunk_text:
                    # Current chunk is full, save it
                    chunked_sections.append(
                        {
                            "id": f"{section['id']}_chunk{chunk_num}",
                            "section_number": section["section_number"],
                            "content": chunk_text.strip(),
                            "chunk": chunk_num,
                            "is_chunked": True,
                        }
                    )
                    chunk_text = sentence
                    chunk_num += 1
                else:
                    chunk_text = test_text

            # Add the last chunk
            if chunk_text:
                chunked_sections.append(
                    {
                        "id": f"{section['id']}_chunk{chunk_num}",
                        "section_number": section["section_number"],
                        "content": chunk_text.strip(),
                        "chunk": chunk_num,
                        "is_chunked": True,
                    }
                )

    return chunked_sections


async def create_replace_index():
    pc = PineconeAsyncio(api_key=PINECONE_API_KEY)
    index_list_response = await pc.list_indexes()
    sem = asyncio.Semaphore(3)

    if index_list_response:

        async def bounded_delete(index_name):
            async with sem:
                await pc.delete_index(index_name)
                print(f"Deleted index: {index_name}")

        tasks = [bounded_delete(index["name"]) for index in index_list_response]
        await asyncio.gather(*tasks)

    async def bounded_create(index_name, model):
        async with sem:
            await pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={"model": model, "field_map": {"text": "content"}},
            )
            print(f"Created new index: {index_name}")

    tasks = [
        bounded_create(PINECONE_DENSE_INDEX_NAME, PINECONE_DENSE_INDEX_MODEL),
        bounded_create(PINECONE_SPARSE_INDEX_NAME, PINECONE_SPARSE_INDEX_MODEL),
    ]
    await asyncio.gather(*tasks)
    
    # Close the Pinecone client
    await pc.close()


def merge_chunks(h1, h2):
    """Get the unique hits from two search results and return them as single array of {'_id', 'chunk_text'} dicts, printing each dict on a new line."""
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


async def query_dense_index_async(query: str, top_k: int = 20):
    """Query the dense index asynchronously."""
    pc_async = PineconeAsyncio(api_key=PINECONE_API_KEY)
    async_dense_index = pc_async.IndexAsyncio(host=PINECONE_DENSE_HOST)
    try:
        return await async_dense_index.search_records(
            namespace="bns_and_bnss",
            query={"top_k": top_k, "inputs": {"text": query}},
        )
    finally:
        await pc_async.close()


async def query_sparse_index_async(query: str, top_k: int = 20):
    """Query the sparse index asynchronously."""
    pc_async = PineconeAsyncio(api_key=PINECONE_API_KEY)
    async_sparse_index = pc_async.IndexAsyncio(host=PINECONE_SPARSE_HOST)
    try:
        return await async_sparse_index.search_records(
            namespace="bns_and_bnss",
            query={"top_k": top_k, "inputs": {"text": query}},
        )
    finally:
        await pc_async.close()


async def query_legal_assistant_async(
    query: str, top_k: int = 20, top_n: int = 10
) -> str:
    """
    Async version: Query the legal assistant with a question and get an answer based on RAG.
    Uses Pinecone's native async methods with IndexAsyncio.

    Args:
        query: The user's legal question
        top_k: Number of results to retrieve from each index
        top_n: Number of results to keep after reranking

    Returns:
        The AI assistant's answer
    """

    pc_async = PineconeAsyncio(api_key=PINECONE_API_KEY)
    async_client = AsyncGroq(api_key=GROQ_API_KEY)

    # Run both searches concurrently using asyncio.gather
    dense_response, sparse_response = await asyncio.gather(
        query_dense_index_async(query, top_k), query_sparse_index_async(query, top_k)
    )

    # Merge results
    merged_results = merge_chunks(sparse_response, dense_response)

    # Rerank results (using asyncio.to_thread for sync operation)
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

    # Create system prompt using langchain-ai/retrieval-qa-chat template
    system_message = f"Answer any use questions based solely on the context below:\n\n<context>\n{combined_context.strip()}\n</context>"

    # Get response from LLM using async client
    chat_completion = await async_client.chat.completions.create(
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
        model=LLM_MODEL_NAME,
    )

    await pc_async.close()

    return chat_completion.choices[0].message.content
