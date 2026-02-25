import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any

import structlog
from fastapi import FastAPI

from app.core.config import get_settings
from app.services.inference import InferenceService

logger = structlog.get_logger(__name__)


class ResourceManager:
    """Manages application resources with proper cleanup."""

    def __init__(self) -> None:
        self._resources: dict[str, Any] = {}
        self._cleanup_tasks: list[asyncio.Task[Any]] = []

    async def startup(self) -> None:
        """Initialize resources on startup."""
        settings = get_settings()
        logger.info("Starting resource initialization")

        inference_device = (
            None if settings.inference_device == "auto" else settings.inference_device
        )
        inference_service = InferenceService(
            artifacts_dir=settings.model_artifacts_dir,
            device=inference_device,
        )

        try:
            inference_service.load()
            self._resources["inference_service"] = inference_service
            logger.info(
                "Inference service initialized",
                artifacts_dir=settings.model_artifacts_dir,
                models=len(inference_service.list_models()),
                device=inference_service.device,
            )
        except Exception as e:
            logger.error(
                "Failed to initialize inference service",
                error=str(e),
                artifacts_dir=settings.model_artifacts_dir,
            )
            if settings.fail_on_missing_artifacts:
                raise

        logger.info("Resource initialization complete")

    async def shutdown(self) -> None:
        logger.info("Starting resource cleanup")

        for task in self._cleanup_tasks:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        for name, resource in list(self._resources.items()):
            try:
                if hasattr(resource, "aclose"):
                    await resource.aclose()
                elif hasattr(resource, "close"):
                    resource.close()
                logger.info("Closed resource", resource=name)
            except Exception as e:
                logger.error("Failed to close resource", resource=name, error=str(e))

        self._resources.clear()
        logger.info("Resource cleanup complete")

    def get_resource(self, name: str) -> Any:
        return self._resources.get(name)


resource_manager = ResourceManager()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    await resource_manager.startup()
    try:
        yield
    finally:
        await resource_manager.shutdown()
