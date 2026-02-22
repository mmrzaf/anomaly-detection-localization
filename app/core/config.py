from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Anomaly Detection and Localization API"
    app_version: str = "0.1.0"
    app_env: Literal["dev", "stg", "prd"] = "dev"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # API
    api_prefix: str = "/api"
    allowed_hosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"])
    request_max_bytes: int = 10_000_000  # 10MB (image uploads)
    # CORS
    cors_origins: list[str] = Field(default_factory=list)
    cors_credentials: bool = False
    cors_methods: list[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: list[str] = Field(default_factory=list)

    # Health check
    health_check_details: bool = True


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
