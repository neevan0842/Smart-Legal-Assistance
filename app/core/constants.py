from enum import Enum


NAMESPACE = "legal_law_documents"


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class DocumentSourceType(str, Enum):
    UPLOAD = "upload"
    GENERATED = "generated"
