import time
from app.core.logger import logger
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Start timing the request
        start_time = time.time()

        # Process the request and get response
        response: Response = await call_next(request)

        # Calculate processing time
        process_time = (time.time() - start_time) * 1000  # in milliseconds

        # Add a custom header for performance monitoring
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

        # Add enhanced security headers
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Log details
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms"
        )

        return response
