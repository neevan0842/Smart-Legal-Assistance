import fitz
import re
import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_DENSE_INDEX_NAME = os.getenv("PINECONE_DENSE_INDEX_NAME")
PINECONE_SPARSE_INDEX_NAME = os.getenv("PINECONE_SPARSE_INDEX_NAME")
PINECONE_DENSE_INDEX_MODEL = os.getenv("PINECONE_DENSE_INDEX_MODEL")
PINECONE_SPARSE_INDEX_MODEL = os.getenv("PINECONE_SPARSE_INDEX_MODEL")

if (
    not PINECONE_API_KEY
    or not PINECONE_DENSE_INDEX_NAME
    or not PINECONE_SPARSE_INDEX_NAME
    or not PINECONE_DENSE_INDEX_MODEL
    or not PINECONE_SPARSE_INDEX_MODEL
):
    raise ValueError("Missing required environment variables.")


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


def create_replace_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_list_response = pc.list_indexes()

    if index_list_response:
        for index in index_list_response:
            pc.delete_index(index["name"])
            print(f"Deleted index: {index['name']}")

    pc.create_index_for_model(
        name=PINECONE_DENSE_INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={"model": PINECONE_DENSE_INDEX_MODEL, "field_map": {"text": "content"}},
    )
    print(f"Created new dense index: {PINECONE_DENSE_INDEX_NAME}")

    pc.create_index_for_model(
        name=PINECONE_SPARSE_INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={"model": PINECONE_SPARSE_INDEX_MODEL, "field_map": {"text": "content"}},
    )
    print(f"Created new sparse index: {PINECONE_SPARSE_INDEX_NAME}")
