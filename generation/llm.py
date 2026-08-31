"""Factory for instantiating LLM backends (Gemini and Groq) with dynamic model inspection."""

import logging
from typing import Optional, List
from langchain_core.language_models.chat_models import BaseChatModel
from config.settings import settings

logger = logging.getLogger(__name__)


def list_available_groq_models(api_key: Optional[str] = None) -> List[str]:
    """Dynamically query Groq API for currently active models available to this API key."""
    key = api_key or settings.GROQ_API_KEY
    default_groq = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    if not key:
        return default_groq

    try:
        from groq import Groq
        client = Groq(api_key=key)
        model_list = client.models.list()
        available = [
            m.id for m in model_list.data 
            if m.active and not "whisper" in m.id.lower() and not "guard" in m.id.lower()
        ]
        # Ensure priority models appear at top
        for prio in reversed(["openai/gpt-oss-120b", "openai/gpt-oss-20b"]):
            if prio in available:
                available.remove(prio)
                available.insert(0, prio)
        return available if available else default_groq
    except Exception as e:
        logger.warning(f"Could not dynamically query Groq models: {e}")
        return default_groq


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Initialize and return a chat LLM instance."""
    selected_provider = (provider or settings.LLM_PROVIDER).lower()

    if selected_provider == "groq":
        key = api_key or settings.GROQ_API_KEY
        if not key:
            raise ValueError(
                "Groq API Key is missing. Please provide it in the sidebar or in your .env file."
            )
        from langchain_groq import ChatGroq

        model = model_name or "openai/gpt-oss-120b"
        logger.info(f"Initializing ChatGroq (model={model}, temperature={temperature})")
        return ChatGroq(
            model_name=model,
            groq_api_key=key,
            temperature=temperature,
        )

    elif selected_provider == "gemini":
        key = api_key or settings.GOOGLE_API_KEY
        if not key:
            raise ValueError(
                "Google Gemini API Key is missing. Please provide it in the sidebar or in your .env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = model_name or "gemini-3.5-flash"
        logger.info(f"Initializing ChatGoogleGenerativeAI (model={model}, temperature={temperature})")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=temperature,
            convert_system_message_to_human=True,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{selected_provider}'. Choose 'groq' or 'gemini'."
        )
