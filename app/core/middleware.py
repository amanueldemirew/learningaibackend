import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import HTTPException, status
from app.core.logger import logger


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle request timeouts.
    """

    def __init__(self, app, timeout_seconds=60):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        try:
            response = await call_next(request)
            return response
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.structured_error(
                f"Request failed after {elapsed_time:.2f} seconds", error=e
            )

            if "timeout" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=f"Request timed out after {elapsed_time:.2f} seconds",
                )
            raise
        finally:
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout_seconds:
                logger.structured_warning(
                    f"Request took {elapsed_time:.2f} seconds (exceeded {self.timeout_seconds}s timeout)"
                )
            else:
                logger.structured_info(
                    f"Request completed in {elapsed_time:.2f} seconds"
                )
