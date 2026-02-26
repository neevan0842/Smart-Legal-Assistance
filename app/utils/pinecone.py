import asyncio
from typing import Dict, List
from pinecone import PineconeAsyncio
from app.core.constants import NAMESPACE
from app.core.logger import logger
from app.core.config import settings
from app.utils.utils import chunk_data


PINECONE_API_KEY = settings.PINECONE_API_KEY
PINECONE_DENSE_HOST = settings.PINECONE_DENSE_HOST
PINECONE_SPARSE_HOST = settings.PINECONE_SPARSE_HOST
PINECONE_DENSE_INDEX_NAME = settings.PINECONE_DENSE_INDEX_NAME
PINECONE_SPARSE_INDEX_NAME = settings.PINECONE_SPARSE_INDEX_NAME
PINECONE_DENSE_INDEX_MODEL = settings.PINECONE_DENSE_INDEX_MODEL
PINECONE_SPARSE_INDEX_MODEL = settings.PINECONE_SPARSE_INDEX_MODEL
PINECONE_RERANKING_MODEL = settings.PINECONE_RERANKING_MODEL


class PineconeService:
    def __init__(
        self,
        pc_async: PineconeAsyncio = None,
        async_dense_index: PineconeAsyncio.IndexAsyncio = None,
        async_sparse_index: PineconeAsyncio.IndexAsyncio = None,
    ):
        self.pc_async = pc_async
        self.async_dense_index = async_dense_index
        self.async_sparse_index = async_sparse_index

    def initialize_clients(self):
        """Initialize Pinecone clients for async operations."""
        self.pc_async = PineconeAsyncio(api_key=PINECONE_API_KEY)
        self.async_dense_index = self.pc_async.IndexAsyncio(host=PINECONE_DENSE_HOST)
        self.async_sparse_index = self.pc_async.IndexAsyncio(host=PINECONE_SPARSE_HOST)

    def get_clients(self):
        """Return the initialized Pinecone clients."""
        if (
            not self.pc_async
            or not self.async_dense_index
            or not self.async_sparse_index
        ):
            self.initialize_clients()
        return self.pc_async, self.async_dense_index, self.async_sparse_index

    async def close_clients(self):
        to_close = []
        if self.async_dense_index:
            to_close.append(self.async_dense_index)
        if self.async_sparse_index:
            to_close.append(self.async_sparse_index)
        if self.pc_async:
            to_close.append(self.pc_async)
        await asyncio.gather(*(client.close() for client in to_close))
        logger.info("All Pinecone clients have been closed.")

    async def create_replace_index(self, semaphore_limit: int = 3):
        """Creates new Pinecone indexes for dense and sparse embeddings, replacing existing ones if they exist."""

        index_list_response = await self.pc_async.list_indexes()
        sem = asyncio.Semaphore(semaphore_limit)

        # Delete existing indexes if they exist
        if index_list_response:

            async def bounded_delete(index_name):
                async with sem:
                    try:
                        await self.pc_async.delete_index(index_name)
                        logger.info(f"Deleted index: {index_name}")
                    except Exception as e:
                        logger.error(f"Error deleting index {index_name}: {e}")

            tasks = [bounded_delete(index["name"]) for index in index_list_response]
            await asyncio.gather(*tasks)

        # Create new indexes with concurrency control
        async def bounded_create(index_name, model):
            async with sem:
                try:
                    await self.pc_async.create_index_for_model(
                        name=index_name,
                        cloud="aws",
                        region="us-east-1",
                        embed={"model": model, "field_map": {"text": "content"}},
                    )
                    logger.info(f"Created new index: {index_name}")
                except Exception as e:
                    logger.error(f"Error creating index {index_name}: {e}")

        tasks = [
            bounded_create(PINECONE_DENSE_INDEX_NAME, PINECONE_DENSE_INDEX_MODEL),
            bounded_create(PINECONE_SPARSE_INDEX_NAME, PINECONE_SPARSE_INDEX_MODEL),
        ]
        await asyncio.gather(*tasks)

    async def check_index_exists(self, index_names: List[str]) -> List[bool]:
        """Check if the specified indexes exist in Pinecone."""
        try:
            existing_indexes = await self.pc_async.list_indexes()
            existing_index_names = {index["name"] for index in existing_indexes}
            return [name in existing_index_names for name in index_names]
        except Exception as e:
            logger.error(f"Error checking index existence: {e}")
            return [False] * len(index_names)

    async def upsert_records(
        self,
        records: List[dict],
        semaphore: int = 5,
        max_batch_size: int = 96,
        namespace: str = NAMESPACE,
    ):
        sem = asyncio.Semaphore(semaphore)

        try:
            total_batches = (len(records) + max_batch_size - 1) // max_batch_size

            async def bounded_upsert(i, index, records, index_name):
                async with sem:
                    await index.upsert_records(namespace, records)
                    logger.info(
                        f"Upserted batch {i+1}/{total_batches} into {index_name} (namespace: {namespace})"
                    )

            tasks = [
                bounded_upsert(
                    i, self.async_dense_index, batch, PINECONE_DENSE_INDEX_NAME
                )
                for i, batch in enumerate(chunk_data(records, max_batch_size))
            ]
            tasks += [
                bounded_upsert(
                    i, self.async_sparse_index, batch, PINECONE_SPARSE_INDEX_NAME
                )
                for i, batch in enumerate(chunk_data(records, max_batch_size))
            ]

            await asyncio.gather(*tasks)
            logger.info("Upserted all records into both indexes.")

        except Exception as e:
            logger.error(f"Error upserting records: {e}")

    async def create_pinecone_namespace(self, namespace: str) -> bool:
        """Create a namespace in both indexes if it doesn't already exist."""
        if not namespace:
            logger.error("Namespace is required for creation.")
            return False

        async def create_namespace_in_sparse_index() -> bool:
            try:
                await self.async_sparse_index.create_namespace(name=namespace)
                return True
            except Exception as e:
                logger.error(f"Error creating namespace in sparse index: {e}")
                return False

        async def create_namespace_in_dense_index() -> bool:
            try:
                await self.async_dense_index.create_namespace(name=namespace)
                return True
            except Exception as e:
                logger.error(f"Error creating namespace in dense index: {e}")
                return False

        results = await asyncio.gather(
            create_namespace_in_sparse_index(), create_namespace_in_dense_index()
        )
        logger.info(f"Create namespace results: {results}")
        return all(results)

    async def delete_document_records_by_metadata(
        self, document_id: str, namespace: str
    ) -> bool:
        """Delete records from both indexes based on document_id in metadata."""
        if not document_id or not namespace:
            logger.error("Document ID and namespace are required for deletion.")
            return False

        async def delete_records_from_sparse_index() -> bool:
            try:
                await self.async_sparse_index.delete(
                    filter={"document_id": {"$eq": document_id}}, namespace=namespace
                )
                return True
            except Exception as e:
                logger.error(f"Error deleting records from sparse index: {e}")
                return False

        async def delete_records_from_dense_index() -> bool:
            try:
                await self.async_dense_index.delete(
                    filter={"document_id": {"$eq": document_id}}, namespace=namespace
                )
                return True
            except Exception as e:
                logger.error(f"Error deleting records from dense index: {e}")
                return False

        results = await asyncio.gather(
            delete_records_from_sparse_index(), delete_records_from_dense_index()
        )
        logger.info(f"Delete results: {results}")
        return all(results)

    async def delete_pinecone_namespace(self, namespace: str) -> bool:
        """Delete an entire namespace from both indexes."""
        if not namespace:
            logger.error("Namespace is required for deletion.")
            return False

        async def delete_namespace_from_sparse_index() -> bool:
            try:
                await self.async_sparse_index.delete_namespace(namespace=namespace)
                return True
            except Exception as e:
                logger.error(f"Error deleting namespace from sparse index: {e}")
                return False

        async def delete_namespace_from_dense_index() -> bool:
            try:
                await self.async_dense_index.delete_namespace(namespace=namespace)
                return True
            except Exception as e:
                logger.error(f"Error deleting namespace from dense index: {e}")
                return False

        results = await asyncio.gather(
            delete_namespace_from_sparse_index(), delete_namespace_from_dense_index()
        )
        logger.info(f"Delete namespace results: {results}")
        return all(results)

    async def rerank_merged_records_and_get_context(
        self,
        query: str,
        merged_results: List[Dict],
        top_n: int = 10,
    ) -> List[str]:
        """Queries both dense and sparse indexes concurrently, merges results, and reranks them using Pinecone's inference API."""

        # Rerank results
        reranked_result = await self.pc_async.inference.rerank(
            model=PINECONE_RERANKING_MODEL,
            query=query,
            documents=merged_results,
            rank_fields=["content"],
            top_n=top_n,
            return_documents=True,
            parameters={"truncate": "END"},
        )

        # Format the combined context for the LLM prompt
        context = []
        for hit in reranked_result.data:
            content = hit["document"]["content"]
            section_number = hit["document"].get("metadata", {}).get("section_number")
            context.append(
                f"Section {section_number} - content: {content}"
                if section_number
                else f"Content: {content}"
            )

        return context

    async def query_index_and_merge(
        self, query: str, top_k: int = 15, namespace: str = NAMESPACE
    ) -> List[Dict]:
        """Query both indexes concurrently and merge results without reranking."""
        dense_response, sparse_response = await asyncio.gather(
            self._query_dense_index_async(query, top_k, namespace=namespace),
            self._query_sparse_index_async(query, top_k, namespace=namespace),
        )

        # Merge results
        merged_results = self._merge_chunks(dense_response, sparse_response)
        return merged_results

    def _merge_chunks(self, h1, h2) -> List[Dict]:
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
                    "section_number": hit["fields"].get("section_number"),
                },
            }
            for hit in sorted_hits
        ]
        return result

    async def _query_dense_index_async(
        self, query: str, top_k: int = 15, namespace: str = NAMESPACE
    ) -> dict:
        """Query the dense index asynchronously."""
        try:
            return await self.async_dense_index.search_records(
                namespace=namespace,
                query={"top_k": top_k, "inputs": {"text": query}},
            )
        except Exception as e:
            logger.error(f"Error querying dense index: {e}")
            return {"result": {"hits": []}}

    async def _query_sparse_index_async(
        self, query: str, top_k: int = 15, namespace: str = NAMESPACE
    ) -> dict:
        """Query the sparse index asynchronously."""
        try:
            return await self.async_sparse_index.search_records(
                namespace=namespace,
                query={"top_k": top_k, "inputs": {"text": query}},
            )
        except Exception as e:
            logger.error(f"Error querying sparse index: {e}")
            return {"result": {"hits": []}}
