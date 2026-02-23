import time

import structlog
from fastapi import Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

logger = structlog.get_logger(__name__)

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds", "HTTP request duration in seconds", ["method", "endpoint"]
)

ERROR_COUNT = Counter(
    "application_errors_total",
    "Total application errors by type and endpoint",
    ["error_type", "endpoint", "status_code"],
)

ACTIVE_REQUESTS = Gauge("http_requests_active", "Number of active HTTP requests")

CRITICAL_ERRORS = Counter("critical_errors_total", "Total number of critical/unhandled errors")

APP_INFO = Info("app_info", "Application information")


class MetricsCollector:
    """Centralized metrics collection with consistent interface."""

    def __init__(self) -> None:
        self.logger = structlog.get_logger(__name__)

    def init_app_info(self, app_name: str, version: str) -> None:
        APP_INFO.info({"name": app_name, "version": version, "component": "fastapi-microservice"})
        self.logger.info("Metrics initialized", app_name=app_name, version=version)

    def record_request(self, method: str, endpoint: str, status_code: int, duration: float) -> None:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

    def record_error(self, error_type: str, endpoint: str, status_code: int | None = None) -> None:
        ERROR_COUNT.labels(
            error_type=error_type,
            endpoint=endpoint,
            status_code=str(status_code) if status_code is not None else "unknown",
        ).inc()

    def record_critical_error(self) -> None:
        CRITICAL_ERRORS.inc()

    def increment_active_requests(self) -> None:
        ACTIVE_REQUESTS.inc()

    def decrement_active_requests(self) -> None:
        ACTIVE_REQUESTS.dec()


metrics = MetricsCollector()


def init_metrics(app_name: str, version: str) -> None:
    metrics.init_app_info(app_name, version)


def _resolve_endpoint_label(request: Request) -> str:
    route = request.scope.get("route")
    path_format = getattr(route, "path_format", None) if route is not None else None
    return str(path_format or request.url.path)


async def metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    metrics.increment_active_requests()
    endpoint = _resolve_endpoint_label(request)

    try:
        response = await call_next(request)
        endpoint = _resolve_endpoint_label(request)
        duration = time.perf_counter() - start_time
        metrics.record_request(
            method=request.method,
            endpoint=endpoint,
            status_code=int(response.status_code),
            duration=duration,
        )
        return response
    except Exception as e:
        endpoint = _resolve_endpoint_label(request)
        duration = time.perf_counter() - start_time
        metrics.record_request(
            method=request.method,
            endpoint=endpoint,
            status_code=500,
            duration=duration,
        )
        metrics.record_error(
            error_type=e.__class__.__name__,
            endpoint=endpoint,
            status_code=500,
        )
        raise
    finally:
        metrics.decrement_active_requests()


def get_metrics() -> tuple[str, str]:
    try:
        data = generate_latest().decode("utf-8")
        return data, CONTENT_TYPE_LATEST
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        return "# Metrics temporarily unavailable\n", CONTENT_TYPE_LATEST
