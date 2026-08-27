"""Factory for instantiating LLM backends (Gemini and OpenAI) with secure parameter handling."""

import logging
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from config.settings import settings

logger = logging.getLogger(__name__)


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Initialize and return a chat LLM instance."""
    selected_provider = (provider or settings.LLM_PROVIDER).lower()

    if selected_provider == "gemini":
        key = api_key or settings.GOOGLE_API_KEY
        if not key:
            raise ValueError(
                "Google Gemini API Key is missing. Please provide it in the sidebar or in your .env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = model_name or settings.LLM_MODEL
        logger.info(f"Initializing ChatGoogleGenerativeAI (model={model}, temperature={temperature})")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=temperature,
            convert_system_message_to_human=True,
        )

    elif selected_provider == "openai":
        key = api_key or settings.OPENAI_API_KEY
        if not key:
            raise ValueError(
                "OpenAI API Key is missing. Please provide it in the sidebar or in your .env file."
            )
        from langchain_openai import ChatOpenAI

        model = model_name or "gpt-4o-mini"
        logger.info(f"Initializing ChatOpenAI (model={model}, temperature={temperature})")
        return ChatOpenAI(
            model=model,
            openai_api_key=key,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{selected_provider}'. Choose 'gemini' or 'openai'."
        )
