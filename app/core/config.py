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
    PINECONE_RERANKING_MODEL: str

    # GROQ CONFIGURATION
    GROQ_API_KEY: str
    LLM_MODEL_NAME: str

    # FRONTEND CONFIGURATION
    FRONTEND_URLS: str

    # Database Configuration
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: int
    POSTGRES_HOST: str
    DATABASE_URL_SYNC: str
    DATABASE_URL_ASYNC: str

    # JWT Configuration
    DUMMY_HASH: str
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int

    model_config = ConfigDict(env_file=".env")


settings = Settings()
