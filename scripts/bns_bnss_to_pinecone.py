import os
from pinecone import Pinecone
from pathlib import Path
from dotenv import load_dotenv
from utils import (
    extract_sections_general,
    save_as_json,
    chunk_data,
    create_replace_index,
    chunk_long_sections,
)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_DENSE_INDEX_NAME = os.getenv("PINECONE_DENSE_INDEX_NAME")
PINECONE_SPARSE_INDEX_NAME = os.getenv("PINECONE_SPARSE_INDEX_NAME")
PINECONE_DENSE_HOST = os.getenv("PINECONE_DENSE_HOST")
PINECONE_SPARSE_HOST = os.getenv("PINECONE_SPARSE_HOST")
MAX_BATCH_SIZE = 96

if (
    not PINECONE_API_KEY
    or not PINECONE_DENSE_INDEX_NAME
    or not PINECONE_SPARSE_INDEX_NAME
    or not PINECONE_DENSE_HOST
    or not PINECONE_SPARSE_HOST
):
    raise ValueError("Missing required Pinecone environment variables.")

pc = Pinecone(api_key=PINECONE_API_KEY)

bns_path = Path.cwd() / "documents" / "BNS.pdf"
bnss_path = Path.cwd() / "documents" / "BNSS.pdf"
output_dir = Path.cwd() / "contents"

bns = extract_sections_general(bns_path.as_posix())
print(f"Extracted BNS sections")

bnss = extract_sections_general(bnss_path.as_posix())
print(f"Extracted BNSS sections")

# Chunk sections that are too large for Pinecone metadata limits
bns_and_bnss = chunk_long_sections(bns + bnss)
save_as_json(bns_and_bnss, output_dir / "bns_and_bnss.json")
print(f"bns_and_bnss saved to JSON with {len(bns_and_bnss)} sections.")

create_replace_index()

if not pc.has_index(PINECONE_DENSE_INDEX_NAME) or not pc.has_index(
    PINECONE_SPARSE_INDEX_NAME
):
    raise ValueError(
        "Required Pinecone indexes do not exist. Please create them first."
    )

dense_index = pc.Index(host=PINECONE_DENSE_HOST)
sparse_index = pc.Index(host=PINECONE_SPARSE_HOST)

for i, batch in enumerate(chunk_data(bns_and_bnss, MAX_BATCH_SIZE)):
    dense_index.upsert_records("bns_and_bnss", batch)
    sparse_index.upsert_records("bns_and_bnss", batch)
    print(f"Upserted batch {i+1} out of {len(bns_and_bnss)//MAX_BATCH_SIZE} records.")

print("Upserted all records into both indexes.")
