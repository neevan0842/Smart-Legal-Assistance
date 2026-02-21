import asyncio
from pathlib import Path
from app.core.config import settings
from app.utils.pinecone import PineconeService
from app.core.logger import logger
from app.utils.utils import chunk_long_sections, save_as_json
from scripts.extraction import extract_bns, extract_bnss


PINECONE_API_KEY = settings.PINECONE_API_KEY
PINECONE_DENSE_INDEX_NAME = settings.PINECONE_DENSE_INDEX_NAME
PINECONE_SPARSE_INDEX_NAME = settings.PINECONE_SPARSE_INDEX_NAME
PINECONE_DENSE_HOST = settings.PINECONE_DENSE_HOST
PINECONE_SPARSE_HOST = settings.PINECONE_SPARSE_HOST
MAX_BATCH_SIZE = 96

# Get the script directory and use it as base for relative paths
script_dir = Path(__file__).parent
bns_path = script_dir / "documents" / "BNS.pdf"
bnss_path = script_dir / "documents" / "BNSS.pdf"
output_dir = script_dir / "contents"

bns = extract_bns(bns_path.as_posix())
save_as_json(bns, output_dir / "bns.json")
logger.info(f"Extracted BNS sections and saved to JSON with {len(bns)} sections.")

bnss = extract_bnss(bnss_path.as_posix())
save_as_json(bnss, output_dir / "bnss.json")
logger.info(f"Extracted BNSS sections and saved to JSON with {len(bnss)} sections.")

# Chunk sections that are too large for Pinecone metadata limits
bns_and_bnss = chunk_long_sections(bns + bnss)
save_as_json(bns_and_bnss, output_dir / "bns_and_bnss.json")
logger.info(f"bns_and_bnss saved to JSON with {len(bns_and_bnss)} sections.")


async def main():
    pc_svc = PineconeService()
    try:
        pc_svc.initialize_clients()
        await pc_svc.create_replace_index()

        if not all(
            await pc_svc.check_index_exists(
                [PINECONE_DENSE_INDEX_NAME, PINECONE_SPARSE_INDEX_NAME]
            )
        ):
            raise ValueError(
                "Required Pinecone indexes do not exist. Please create them first."
            )

        await pc_svc.upsert_records(bns_and_bnss)
    finally:
        await pc_svc.close_clients()


if __name__ == "__main__":
    asyncio.run(main())
