import math
from langchain_core.messages import BaseMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import fitz


def save_as_json(data, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def chunk_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def create_system_message(context: str) -> str:
    """Create system prompt with context."""
    return f"""You are a helpful legal assistant. Use only the following context to answer the question. 
    If you don't know the answer, say you don't know. 
    Always use all relevant information from the context to provide a complete and accurate answer.
    \n\n<context>\n{context.strip()}\n</context>"""


def split_text_to_chunks(text: str) -> list[str]:
    """
    Split a large body of text into chunks with overlap
    so semantic context is preserved.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    chunks = text_splitter.split_text(text)
    return chunks


def base_messages_to_groq(messages: list[BaseMessage]) -> list[dict]:
    """
    Convert LangChain BaseMessage objects to a list of dicts
    compatible with Groq's chat API (role must be 'system', 'user', or 'assistant').
    """
    groq_msgs = []

    for msg in messages:
        # Determine the correct role mapping
        if msg.type == "system":
            role = "system"
        elif msg.type == "human" or msg.type == "user":
            role = "user"
        else:
            # Map all LangChain assistant/AI roles to "assistant"
            role = "assistant"

        groq_msgs.append({"role": role, "content": msg.content})

    return groq_msgs


def parse_relevance_score(raw_score):
    """
    Checks if the score is a single string of digits (e.g., '01001200') and converts it to a list of integers.
    Otherwise, returns the score unchanged.
    """
    if isinstance(raw_score, list) and len(raw_score) == 1:
        s = raw_score[0]
        if isinstance(s, str) and s.isdigit():
            return [int(c) for c in s]
        if isinstance(s, int):
            return [int(c) for c in str(s)]
    return raw_score


def get_ndcg_score_at_k(relevance_scores: list[int], k: int) -> float:
    """
    Calculate NDCG@k for a list of relevance scores.
    relevance_scores: List of relevance scores for retrieved documents, ordered by rank.
    k: The rank position to calculate NDCG at.
    """
    k = min(k, len(relevance_scores))

    # DCG@k
    def dcg_at_k(relevance, k):
        k = min(k, len(relevance))
        return sum(
            rel / math.log2(i + 1) for i, rel in enumerate(relevance[:k], start=1)
        )

    # IDCG@k
    def idcg_at_k(relevance, k):
        ideal_relevance = sorted(relevance, reverse=True)
        return dcg_at_k(ideal_relevance, k)

    # NDCG@k
    def ndcg_at_k(relevance, k):
        dcg = dcg_at_k(relevance, k)
        idcg = idcg_at_k(relevance, k)
        return dcg / idcg if idcg > 0 else 0.0

    return ndcg_at_k(relevance_scores, k)


class PDFLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
