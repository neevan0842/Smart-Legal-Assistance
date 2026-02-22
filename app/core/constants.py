from enum import Enum


NAMESPACE = "legal_law_documents"


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
