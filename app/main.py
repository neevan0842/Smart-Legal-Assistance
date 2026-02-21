from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.utils.groq import GroqService
from app.utils.pinecone import PineconeService
from app.core.config import settings
from app.api import generate, auth
from app.middlewares.logger import LoggingMiddleware

FRONTEND_URLS = settings.FRONTEND_URLS


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pinecone_service = PineconeService()
    app.state.groq_service = GroqService()
    app.state.pinecone_service.initialize_clients()
    app.state.groq_service.initialize_client()

    yield

    await app.state.pinecone_service.close_clients()
    await app.state.groq_service.close_client()


app = FastAPI(lifespan=lifespan)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

app.include_router(router=generate.router, prefix="/api")
app.include_router(router=auth.router, prefix="/api")


# Root route
@app.get("/")
async def root():
    return {"message": "Hello World"}


# Healthcheck route
@app.get("/healthcheck", tags=["healthcheck"])
async def healthcheck():
    return {"message": "ok"}


# Exception handler example
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )
