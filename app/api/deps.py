from app.core.config import Settings, get_settings
from app.core.lifecycle import resource_manager
from app.services.inference import InferenceService


def get_current_settings() -> Settings:
    """Dependency to get current settings."""
    return get_settings()


def get_resource_manager() -> object:
    """Dependency to get resource manager."""
    return resource_manager


def get_inference_service() -> InferenceService:
    service = resource_manager.get_resource("inference_service")
    if service is None:
        raise RuntimeError("Inference service is not initialized")
    if not isinstance(service, InferenceService):
        raise RuntimeError("Invalid inference service in resource manager")
    return service
