import asyncio
from pinecone import PineconeAsyncio
from pathlib import Path
from scripts.utils import (
    extract_sections_general,
    save_as_json,
    chunk_data,
    create_replace_index,
    chunk_long_sections,
)
from app.core.config import settings


PINECONE_API_KEY = settings.PINECONE_API_KEY
PINECONE_DENSE_INDEX_NAME = settings.PINECONE_DENSE_INDEX_NAME
PINECONE_SPARSE_INDEX_NAME = settings.PINECONE_SPARSE_INDEX_NAME
PINECONE_DENSE_HOST = settings.PINECONE_DENSE_HOST
PINECONE_SPARSE_HOST = settings.PINECONE_SPARSE_HOST
MAX_BATCH_SIZE = 96


pc_async = PineconeAsyncio(api_key=PINECONE_API_KEY)

# Get the script directory and use it as base for relative paths
script_dir = Path(__file__).parent
bns_path = script_dir / "documents" / "BNS.pdf"
bnss_path = script_dir / "documents" / "BNSS.pdf"
output_dir = script_dir / "contents"

bns = extract_sections_general(bns_path.as_posix())
print(f"Extracted BNS sections")

bnss = extract_sections_general(bnss_path.as_posix())
print(f"Extracted BNSS sections")

# Chunk sections that are too large for Pinecone metadata limits
bns_and_bnss = chunk_long_sections(bns + bnss)
save_as_json(bns_and_bnss, output_dir / "bns_and_bnss.json")
print(f"bns_and_bnss saved to JSON with {len(bns_and_bnss)} sections.")


async def main():
    await create_replace_index()

    if not await pc_async.has_index(
        PINECONE_DENSE_INDEX_NAME
    ) or not await pc_async.has_index(PINECONE_SPARSE_INDEX_NAME):
        raise ValueError(
            "Required Pinecone indexes do not exist. Please create them first."
        )

    dense_index = pc_async.IndexAsyncio(host=PINECONE_DENSE_HOST)
    sparse_index = pc_async.IndexAsyncio(host=PINECONE_SPARSE_HOST)

    sem = asyncio.Semaphore(5)  # Limit concurrent upserts to avoid rate limits

    try:
        total_batches = (len(bns_and_bnss) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE

        async def bounded_upsert(i, index, records, index_name):
            async with sem:
                await index.upsert_records("bns_and_bnss", records)
                print(
                    f"Upserted batch {i+1} out of {total_batches} records into {index_name}."
                )

        tasks = [
            bounded_upsert(i, dense_index, batch, PINECONE_DENSE_INDEX_NAME)
            for i, batch in enumerate(chunk_data(bns_and_bnss, MAX_BATCH_SIZE))
        ]
        tasks += [
            bounded_upsert(i, sparse_index, batch, PINECONE_SPARSE_INDEX_NAME)
            for i, batch in enumerate(chunk_data(bns_and_bnss, MAX_BATCH_SIZE))
        ]
        await asyncio.gather(*tasks)

        print("Upserted all records into both indexes.")
    finally:
        await dense_index.close()
        await sparse_index.close()
        await pc_async.close()


if __name__ == "__main__":
    asyncio.run(main())
