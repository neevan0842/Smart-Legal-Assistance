from enum import Enum
from pathlib import Path


NAMESPACE = "legal_law_documents"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
USER_UPLOADS_DIR = STORAGE_DIR / "user_uploads"
GENERATED_DOCS_DIR = STORAGE_DIR / "generated_docs"
USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DOCS_DIR.mkdir(parents=True, exist_ok=True)


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class DocumentSourceType(str, Enum):
    UPLOAD = "upload"
    GENERATED = "generated"
