import json
import re
from typing import Dict, List


def save_as_json(data, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def chunk_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def chunk_long_sections(
    sections: List[Dict], max_content_bytes: int = 20000
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
