import os
from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Model Configuration
    MODEL_NAME: str = Field(default="yolo11n.pt", description="YOLO model checkpoint name or path")
    CONFIDENCE_THRESHOLD: float = Field(default=0.40, ge=0.01, le=1.0)
    IOU_THRESHOLD: float = Field(default=0.45, ge=0.01, le=1.0)
    
    # Confidence Tier Thresholds
    CONFIDENCE_HIGH_THRESHOLD: float = Field(default=0.80, ge=0.50, le=1.0)
    CONFIDENCE_MEDIUM_THRESHOLD: float = Field(default=0.60, ge=0.30, le=0.90)

    # Image constraints
    MAX_IMAGE_SIZE_MB: int = Field(default=15)
    ALLOWED_IMAGE_TYPES: List[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    )
    MAX_IMAGE_DIMENSION: int = Field(default=4096)

    # LLM Settings
    LLM_PROVIDER: str = Field(default="groq", description="gemini, openai, groq, ollama, or mock")
    LLM_MODEL: str = Field(default="openai/gpt-oss-120b")
    ENABLE_VISION_LLM: bool = Field(default=True)
    
    # API Keys
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    GROQ_API_KEY: Optional[str] = Field(default=None)
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None)
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")

    # Guardrails
    MAX_QUESTION_LENGTH: int = Field(default=300)
    ENABLE_GUARDRAILS: bool = Field(default=True)

    # Authentication & Security
    AUTH_ENABLED: bool = Field(default=True)
    SESSION_TIMEOUT_MINUTES: int = Field(default=60)
    MAX_LOGIN_ATTEMPTS: int = Field(default=5)
    DATABASE_URL: Optional[str] = Field(default=None)
    APP_ENV: str = Field(default="development")

    # Observability
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default="logs/vision_agent.log")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
