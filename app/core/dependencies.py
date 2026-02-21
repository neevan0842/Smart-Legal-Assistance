from fastapi import Request


def get_pinecone_service(request: Request):
    return request.app.state.pinecone_service


def get_groq_service(request: Request):
    return request.app.state.groq_service
