from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Application
    quiz_secret: str
    log_level: str = "INFO"
    force_submit_time: int = 170

    # LLM Provider
    llm_provider: Literal["GEMINI", "AIPIPE"] = "GEMINI"
    llm_fallback_enabled: bool = True

    # Google Gemini Direct
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash-lite"

    # AIpipe
    aipipe_api_key: str | None = None
    aipipe_model: str = "gemini-2.5-flash-lite"

    # Browser
    playwright_timeout: int = 30000
    headless: bool = True


# Global settings instance
settings = Settings()
