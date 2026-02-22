import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any

import structlog
from fastapi import FastAPI

logger = structlog.get_logger(__name__)


class ResourceManager:
    """Manages application resources with proper cleanup."""

    def __init__(self) -> None:
        self._resources: dict[str, Any] = {}
        self._cleanup_tasks: list[asyncio.Task[Any]] = []

    async def startup(self) -> None:
        """Initialize resources on startup."""
        logger.info("Starting resource initialization")

        # Example: Initialize HTTP client session
        # self._resources["http_session"] = httpx.AsyncClient()

        logger.info("Resource initialization complete")

    async def shutdown(self) -> None:
        """Clean up resources on shutdown."""
        logger.info("Starting resource cleanup")

        # Cancel any running tasks
        for task in self._cleanup_tasks:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        # Close resources
        for name, resource in self._resources.items():
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
        """Get a managed resource."""
        return self._resources.get(name)


# Global resource manager instance
resource_manager = ResourceManager()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    await resource_manager.startup()

    try:
        yield
    finally:
        # Shutdown
        await resource_manager.shutdown()
