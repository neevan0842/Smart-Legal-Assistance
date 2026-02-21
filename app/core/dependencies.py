from fastapi import Request
from app.db.session import AsyncSessionLocal


def get_pinecone_service(request: Request):
    return request.app.state.pinecone_service


def get_groq_service(request: Request):
    return request.app.state.groq_service


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
