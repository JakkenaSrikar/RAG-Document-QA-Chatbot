"""Centralized application settings with Streamlit Cloud secrets fallback."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


def get_secret(key: str, default: str = "") -> str:
    """Helper to fetch config from env, os.environ, or streamlit secrets."""
    val = os.getenv(key, "")
    if not val:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and key in st.secrets:
                val = str(st.secrets[key])
        except Exception:
            pass
    return val or default


class Settings(BaseSettings):
    """Application configuration loaded from environment variables, .env, or Streamlit secrets."""

    # API Keys
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # Provider Selections
    LLM_PROVIDER: str = "groq"  # "groq" or "gemini"
    LLM_MODEL: str = "openai/gpt-oss-120b"
    EMBEDDING_PROVIDER: str = "huggingface"  # "huggingface" or "gemini"
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
# Fallback to secrets if env values are empty
if not settings.GOOGLE_API_KEY:
    settings.GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
if not settings.GROQ_API_KEY:
    settings.GROQ_API_KEY = get_secret("GROQ_API_KEY")

settings.ensure_directories()
