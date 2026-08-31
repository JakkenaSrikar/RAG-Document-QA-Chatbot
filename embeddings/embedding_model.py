"""Factory to initialize embedding models with fallback support."""

import logging
from typing import Optional
from langchain_core.embeddings import Embeddings
from config.settings import settings

logger = logging.getLogger(__name__)


def get_embedding_model(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Embeddings:
    """Instantiate and return the appropriate embedding model based on provider."""
    selected_provider = (provider or settings.EMBEDDING_PROVIDER).lower()

    if selected_provider == "gemini":
        key = api_key or settings.GOOGLE_API_KEY
        if not key:
            raise ValueError("Google API Key is required for Gemini embeddings.")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        # Verified active embedding models on Gemini API: models/gemini-embedding-001 or models/gemini-embedding-2
        embed_model = model_name or "models/gemini-embedding-001"
        logger.info(f"Initializing Google Gemini Embeddings ({embed_model})")
        return GoogleGenerativeAIEmbeddings(
            model=embed_model,
            google_api_key=key,
        )

    elif selected_provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embed_model = model_name or settings.HUGGINGFACE_MODEL
        logger.info(f"Initializing HuggingFace Embeddings ({embed_model})")
        return HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    else:
        raise ValueError(
            f"Unsupported embedding provider: '{selected_provider}'. Choose 'gemini' or 'huggingface'."
        )
