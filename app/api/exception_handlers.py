from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    BaseAppException,
    ConflictError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from app.core.logging import get_logger
from app.core.metrics import metrics

logger = get_logger(__name__)


class ErrorResponseBuilder:
    """Builds consistent error responses with proper context."""

    @staticmethod
    def build_error_response(
        request: Request,
        message: str,
        code: str,
        error_type: str,
        details: dict[str, Any] | None = None,
        status_code: int = 400,
    ) -> dict[str, Any]:
        """Build standardized error response."""
        request_id = getattr(request.state, "request_id", "unknown")

        return {
            "error": {
                "message": message,
                "code": code,
                "type": error_type,
                "details": details or {},
            },
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path,
            "method": request.method,
            "status_code": status_code,
        }

    @staticmethod
    def get_response_headers(exc: Exception) -> dict[str, str]:
        """Get appropriate response headers for exception type."""
        headers = {}

        if isinstance(exc, RateLimitError):
            headers["Retry-After"] = "60"
            headers["X-RateLimit-Limit"] = "100"
            headers["X-RateLimit-Remaining"] = "0"
        elif isinstance(exc, AuthenticationError):
            headers["WWW-Authenticate"] = 'Bearer realm="api"'

        return headers


async def base_app_exception_handler(request: Request, exc: BaseAppException) -> JSONResponse:
    """
    Handle custom application exceptions with proper metrics and audit logging.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    user_id = getattr(request.state, "user_id", None)

    status_code_map = {
        AuthenticationError: status.HTTP_401_UNAUTHORIZED,
        AuthorizationError: status.HTTP_403_FORBIDDEN,
        NotFoundError: status.HTTP_404_NOT_FOUND,
        ConflictError: status.HTTP_409_CONFLICT,
        ValidationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        RateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
    }

    status_code = status_code_map.get(type(exc), status.HTTP_400_BAD_REQUEST)

    metrics.record_error(
        error_type=exc.__class__.__name__, endpoint=request.url.path, status_code=status_code
    )

    error_response = ErrorResponseBuilder.build_error_response(
        request=request,
        message=exc.message,
        code=exc.code,
        error_type=exc.__class__.__name__,
        details=exc.details,
        status_code=status_code,
    )

    log_context = {
        "exception_type": exc.__class__.__name__,
        "message": exc.message,
        "code": exc.code,
        "status_code": status_code,
        "request_id": request_id,
        "path": request.url.path,
        "method": request.method,
        "user_id": user_id,
        "details": exc.details,
    }

    if status_code >= 500:
        logger.error("Application exception occurred", **log_context)
    else:
        logger.warning("Client error occurred", **log_context)

    headers = ErrorResponseBuilder.get_response_headers(exc)

    return JSONResponse(status_code=status_code, content=error_response, headers=headers)


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle standard HTTP exceptions with consistent formatting and metrics.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    user_id = getattr(request.state, "user_id", None)

    metrics.record_error(
        error_type="HTTPException", endpoint=request.url.path, status_code=exc.status_code
    )

    error_response = ErrorResponseBuilder.build_error_response(
        request=request,
        message=exc.detail,
        code=f"HTTP_{exc.status_code}",
        error_type="HTTPException",
        status_code=exc.status_code,
    )

    log_context = {
        "status_code": exc.status_code,
        "detail": exc.detail,
        "request_id": request_id,
        "path": request.url.path,
        "method": request.method,
        "user_id": user_id,
    }

    if exc.status_code >= 500:
        logger.error("HTTP server error", **log_context)
    else:
        logger.warning("HTTP client error", **log_context)

    return JSONResponse(status_code=exc.status_code, content=error_response)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions with comprehensive logging and metrics.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    user_id = getattr(request.state, "user_id", None)

    metrics.record_error(
        error_type="UnhandledException", endpoint=request.url.path, status_code=500
    )
    metrics.record_critical_error()

    error_response = ErrorResponseBuilder.build_error_response(
        request=request,
        message="An internal server error occurred. Please try again later.",
        code="INTERNAL_SERVER_ERROR",
        error_type="InternalServerError",
        status_code=500,
    )

    logger.error(
        "Unhandled exception occurred",
        exception_type=exc.__class__.__name__,
        message=str(exc),
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        user_id=user_id,
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response,
    )


async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """
    Handle validation errors with detailed field information.
    Enhanced to provide more context for API consumers.
    """
    # Add validation context to details if not present
    if not exc.details and hasattr(exc, "errors"):
        exc.details = {"validation_errors": exc.errors}

    return await base_app_exception_handler(request, exc)


async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    """
    Handle not found errors with resource context.
    """
    return await base_app_exception_handler(request, exc)


async def request_validation_handler(request, exc: RequestValidationError):
    details = {"validation_errors": exc.errors()}
    wrapped = ValidationError(details=details)  # your app error
    return await base_app_exception_handler(request, wrapped)


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponseBuilder.build_error_response(
            request=request,
            message=str(exc),
            code="VALUE_ERROR",
            error_type="ValueError",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        ),
    )


exception_handlers = {
    BaseAppException: base_app_exception_handler,
    StarletteHTTPException: http_exception_handler,
    ValidationError: validation_error_handler,
    NotFoundError: not_found_handler,
    RequestValidationError: request_validation_handler,
    ValueError: value_error_handler,
    Exception: global_exception_handler,
}


def install_error_handlers(app: FastAPI):
    for exc_type, handler in exception_handlers.items():
        app.add_exception_handler(exc_type, handler)
