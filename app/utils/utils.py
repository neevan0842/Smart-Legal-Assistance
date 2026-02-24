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


class PDFLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
