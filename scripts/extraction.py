import re
import fitz
from typing import Dict, List


def extract_bns(pdf_path: str) -> List[Dict]:
    """Extract sections from BNS PDF based on section numbering patterns."""
    doc = fitz.open(pdf_path)
    try:
        return extract_bns_bnss_common(doc, pdf_path)
    finally:
        doc.close()


def extract_bnss(pdf_path: str) -> List[Dict]:
    """Extract sections from BNSS PDF based on section numbering patterns."""
    result = []
    try:
        doc_part1 = fitz.open(pdf_path)
        doc_part1.select(range(0, 157))
        result_part1 = extract_bns_bnss_common(doc_part1, pdf_path)
        result.extend(result_part1)
        return result
    finally:
        doc_part1.close()


def extract_bns_bnss_common(doc: fitz.Document, pdf_path: str) -> List[Dict]:
    """Extract sections from BNS or BNSS PDF based on section numbering patterns."""

    # --- 1) Read full PDF text by lines ---
    lines = []
    for page in doc:
        text = page.get_text("text")
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("_", "")
        lines.extend(text.split("\n"))

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
                "id": f"{'bnss_p1' if 'bnss' in pdf_path.lower() else 'bns'}_{num}",
                "section_number": num,
                "content": content,
            }
        )

    return result
