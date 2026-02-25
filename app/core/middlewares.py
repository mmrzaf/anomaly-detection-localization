import uuid

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp

from app.core.config import get_settings
from app.core.metrics import metrics_middleware


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = rid
        structlog.contextvars.bind_contextvars(
            request_id=rid,
            path=request.url.path,
            method=request.method,
        )
        try:
            response: Response = await call_next(request)
            response.headers["x-request-id"] = rid
            response.headers["server"] = "app"
            return response
        finally:
            structlog.contextvars.clear_contextvars()


def install_middleware(app: FastAPI):
    settings = get_settings()

    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
        allow_credentials=settings.cors_credentials,
    )
    app.add_middleware(RequestContextMiddleware)

    @app.middleware("http")
    async def enforce_content_length_limit(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH"} and request.url.path != "/openapi.json":
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > settings.request_max_bytes:
                        return Response("Payload too large", status_code=413)
                except ValueError:
                    pass
        return await call_next(request)

    app.middleware("http")(metrics_middleware)
