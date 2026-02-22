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

    def __init__(self):
        self.logger = structlog.get_logger(__name__)

    def init_app_info(self, app_name: str, version: str) -> None:
        """Initialize application information metrics."""
        APP_INFO.info({"name": app_name, "version": version, "component": "fastapi-microservice"})
        self.logger.info("Metrics initialized", app_name=app_name, version=version)

    def record_request(self, method: str, endpoint: str, status_code: int, duration: float) -> None:
        """Record HTTP request metrics."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()

        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

    def record_error(self, error_type: str, endpoint: str, status_code: int | None = None) -> None:
        """Record application error metrics."""
        ERROR_COUNT.labels(
            error_type=error_type,
            endpoint=endpoint,
            status_code=str(status_code) if status_code else "unknown",
        ).inc()

        self.logger.debug(
            "Error metric recorded",
            error_type=error_type,
            endpoint=endpoint,
            status_code=status_code,
        )

    def record_critical_error(self) -> None:
        """Record critical/unhandled error."""
        CRITICAL_ERRORS.inc()

    def increment_active_requests(self) -> None:
        """Increment active request counter."""
        ACTIVE_REQUESTS.inc()

    def decrement_active_requests(self) -> None:
        """Decrement active request counter."""
        ACTIVE_REQUESTS.dec()


metrics = MetricsCollector()


def init_metrics(app_name: str, version: str) -> None:
    """Initialize application metrics."""
    metrics.init_app_info(app_name, version)


async def metrics_middleware(request: Request, call_next):
    """Enhanced middleware to collect comprehensive request metrics."""
    start_time = time.time()

    metrics.increment_active_requests()

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        metrics.record_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=request.scope["route"].path_format,
            duration=duration,
        )

        return response

    except Exception as e:
        duration = time.time() - start_time

        metrics.record_request(
            method=request.method,
            endpoint=request.scope["route"].path_format,
            status_code=500,
            duration=duration,
        )

        metrics.record_error(
            error_type=e.__class__.__name__,
            endpoint=request.scope["route"].path_format,
            status_code=500,
        )

        raise

    finally:
        metrics.decrement_active_requests()


def get_metrics() -> tuple[str, str]:
    """
    Returns metrics data and content type.
    Keeps API completely agnostic of Prometheus.
    """
    try:
        data = generate_latest().decode("utf-8")
        return data, CONTENT_TYPE_LATEST
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        return "# Metrics temporarily unavailable\n", CONTENT_TYPE_LATEST
