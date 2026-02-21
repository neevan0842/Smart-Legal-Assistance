from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    # PINECONE CONFIGURATION
    PINECONE_API_KEY: str
    PINECONE_DENSE_INDEX_NAME: str
    PINECONE_SPARSE_INDEX_NAME: str
    PINECONE_DENSE_INDEX_MODEL: str
    PINECONE_SPARSE_INDEX_MODEL: str
    PINECONE_DENSE_HOST: str
    PINECONE_SPARSE_HOST: str

    # GROQ CONFIGURATION
    GROQ_API_KEY: str
    LLM_MODEL_NAME: str

    # FRONTEND CONFIGURATION
    FRONTEND_URLS: str

    model_config = ConfigDict(env_file=".env")


settings = Settings()
