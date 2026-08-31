"""Centralized application settings using Pydantic Settings."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env."""

    # API Keys
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # Provider Selections
    LLM_PROVIDER: str = "gemini"  # "gemini" or "groq"
    LLM_MODEL: str = "gemini-3.5-flash"
    EMBEDDING_PROVIDER: str = "gemini"  # "gemini" or "huggingface"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    HUGGINGFACE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector Storage Paths
    CHROMA_PERSIST_DIRECTORY: str = str(BASE_DIR / "data" / "chroma")
    UPLOAD_DIRECTORY: str = str(BASE_DIR / "data" / "uploads")
    COLLECTION_NAME: str = "document_qa"

    # Chunking & Retrieval Parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    RETRIEVAL_K: int = 4

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self):
        """Ensure required data directories exist."""
        os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs(self.UPLOAD_DIRECTORY, exist_ok=True)


settings = Settings()
settings.ensure_directories()
