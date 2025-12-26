from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .core.config import settings
from .middlewares.logger import LoggingMiddleware
from pinecone import PineconeAsyncio

FRONTEND_URLS = settings.FRONTEND_URLS
PINECONE_API_KEY = settings.PINECONE_API_KEY
PINECONE_DENSE_HOST = settings.PINECONE_DENSE_HOST
PINECONE_SPARSE_HOST = settings.PINECONE_SPARSE_HOST


@asynccontextmanager
async def lifespan(app: FastAPI):
    pc_async = PineconeAsyncio(api_key=PINECONE_API_KEY)
    async_dense_index = pc_async.IndexAsyncio(host=PINECONE_DENSE_HOST)
    async_sparse_index = pc_async.IndexAsyncio(host=PINECONE_SPARSE_HOST)

    yield

    await async_dense_index.close()
    await async_sparse_index.close()


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

# app.include_router(router=auth.router, prefix="/api/v1")


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
