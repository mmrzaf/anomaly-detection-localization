import platform
from datetime import UTC, datetime
from typing import Any

import psutil
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_current_settings, get_resource_manager
from app.core.config import Settings
from app.core.lifecycle import ResourceManager

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    details: dict[str, Any] | None = None


@router.get("/", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_current_settings),
    rm: ResourceManager = Depends(get_resource_manager),  # type: ignore[assignment]
) -> HealthResponse:
    response = HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC),
        version=settings.app_version,
    )

    if settings.health_check_details:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        inference_service = None
        try:
            inference_service = rm.get_resource("inference_service") if rm is not None else None
        except Exception:
            inference_service = None

        response.details = {
            "service": {
                "name": settings.app_name,
                "debug_mode": settings.debug,
                "log_level": settings.log_level,
                "env": settings.app_env,
            },
            "inference": {
                "artifacts_dir": settings.model_artifacts_dir,
                "loaded": bool(getattr(inference_service, "is_loaded", False)),
                "model_count": len(inference_service.list_models())
                if inference_service is not None
                else 0,
                "device": getattr(inference_service, "device", None),
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
