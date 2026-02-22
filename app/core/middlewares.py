import uuid

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp

from app.core.config import get_settings


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request, call_next):
        rid = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = rid
        structlog.contextvars.bind_contextvars(
            request_id=rid, path=request.url.path, method=request.method
        )
        try:
            return await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()


def install_middleware(app: FastAPI):
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=get_settings().allowed_hosts)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_methods=get_settings().cors_methods,
        allow_headers=get_settings().cors_headers,
        allow_credentials=get_settings().cors_credentials,
    )

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id", str(uuid.uuid4()))
        response: Response = await call_next(request)
        response.headers["x-request-id"] = rid
        response.headers["server"] = "app"  # tame the banner
        return response

    @app.middleware("http")
    async def max_body(request: Request, call_next):
        if request.url.path not in ("/docs", "/openapi.json"):
            body = await request.body()
            if len(body) > get_settings().request_max_bytes:
                return Response("Payload too large", status_code=413)
            request._body = body  # avoid double-read
        return await call_next(request)
