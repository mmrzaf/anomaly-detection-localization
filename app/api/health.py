import platform
from datetime import UTC, datetime
from typing import Any

import psutil
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_current_settings
from app.core.config import Settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    version: str
    details: dict[str, Any] | None = None


@router.get("/", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_current_settings),
) -> HealthResponse:
    """
    Health check endpoint.

    Returns basic service health information and optionally detailed system metrics.
    """
    response = HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC),
        version=settings.app_version,
    )

    if settings.health_check_details:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        response.details = {
            "service": {
                "name": settings.app_name,
                "debug_mode": settings.debug,
                "log_level": settings.log_level,
            },
            "system": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent_used": memory.percent,
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent_used": round((disk.used / disk.total) * 100, 2),
                },
            },
        }

    return response
