from app.core.config import Settings, get_settings
from app.core.lifecycle import resource_manager


def get_current_settings() -> Settings:
    """Dependency to get current settings."""
    return get_settings()


def get_resource_manager() -> object:
    """Dependency to get resource manager."""
    return resource_manager
