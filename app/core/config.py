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
    app_version: str = "0.2.0"
    app_env: Literal["dev", "stg", "prd"] = "dev"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # API
    api_prefix: str = "/api"
    allowed_hosts: list[str] = Field(default_factory=lambda: ["*"])
    request_max_bytes: int = 10_000_000  # 10MB (image uploads)

    # CORS
    cors_origins: list[str] = Field(default_factory=list)
    cors_credentials: bool = False
    cors_methods: list[str] = Field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
    cors_headers: list[str] = Field(default_factory=lambda: ["*"])

    # Health check
    health_check_details: bool = True

    # Model artifacts / inference
    model_artifacts_dir: str = "artifacts"
    inference_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    fail_on_missing_artifacts: bool = True

    # UI / API behavior
    default_return_visuals: bool = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
