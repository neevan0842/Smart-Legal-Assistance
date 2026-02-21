from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import AsyncGroq
from .core.config import settings
from .api.generate import api as generate
from .middlewares.logger import LoggingMiddleware
from pinecone import PineconeAsyncio

FRONTEND_URLS = settings.FRONTEND_URLS
PINECONE_API_KEY = settings.PINECONE_API_KEY
PINECONE_DENSE_HOST = settings.PINECONE_DENSE_HOST
PINECONE_SPARSE_HOST = settings.PINECONE_SPARSE_HOST
GROQ_API_KEY = settings.GROQ_API_KEY


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pc_async = PineconeAsyncio(api_key=PINECONE_API_KEY)
    app.state.async_dense_index = app.state.pc_async.IndexAsyncio(
        host=PINECONE_DENSE_HOST
    )
    app.state.async_sparse_index = app.state.pc_async.IndexAsyncio(
        host=PINECONE_SPARSE_HOST
    )
    app.state.groq_client = AsyncGroq(api_key=GROQ_API_KEY)

    yield

    await app.state.async_dense_index.close()
    await app.state.async_sparse_index.close()
    await app.state.groq_client.close()
    await app.state.pc_async.close()


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


# Root route
@app.get("/")
async def root():
    return {"message": "Hello World"}


# Healthcheck route
@app.get("/healthcheck", tags=["healthcheck"])
async def healthcheck():
    return {"message": "API Working"}


# Exception handler example
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )
