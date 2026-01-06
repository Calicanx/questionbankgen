"""Configuration management for QuestionBank Generator."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class MongoDBConfig(BaseModel):
    """MongoDB configuration."""

    uri: str = Field(default_factory=lambda: os.getenv("MONGODB_URI", ""))
    db_name: str = Field(default_factory=lambda: os.getenv("MONGODB_DB_NAME", "ai_tutor"))
    source_collection: str = "scraped_questions"
    output_collection: str = "generated_questions"


class GeminiConfig(BaseModel):
    """Google Gemini API configuration."""

    api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    image_model: str = Field(default_factory=lambda: os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview"))
    max_retries: int = 3
    temperature: float = 0.7


class ValidationConfig(BaseModel):
    """Validation configuration."""

    strict_mode: bool = True
    verify_answers: bool = True
    check_latex: bool = True



class GCSConfig(BaseModel):
    """Google Cloud Storage configuration."""

    bucket_name: str = Field(default_factory=lambda: os.getenv("GCS_BUCKET_NAME", ""))
    credentials: str = Field(default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))


class Config(BaseModel):
    """Main configuration container."""

    mongodb: MongoDBConfig = Field(default_factory=MongoDBConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    gcs: GCSConfig = Field(default_factory=GCSConfig)
    gcs_credentials: str = Field(default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))

    def validate_required(self) -> list[str]:
        """Validate required configuration values are set."""
        errors = []

        if not self.mongodb.uri:
            errors.append("MONGODB_URI is not set in environment")
        if not self.gemini.api_key:
            errors.append("GEMINI_API_KEY is not set in environment")
        if not self.gcs.bucket_name:
            errors.append("GCS_BUCKET_NAME is not set in environment")
        if not self.gcs_credentials:
            errors.append("GOOGLE_APPLICATION_CREDENTIALS is not set in environment")

        return errors


# Global configuration instance
config = Config()
