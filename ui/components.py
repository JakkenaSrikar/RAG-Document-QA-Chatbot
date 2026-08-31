"""Reusable UI helper components for Streamlit interface with verified models."""

from typing import List, Dict, Any, Callable
import streamlit as st
from generation.llm import list_available_groq_models
from config.settings import settings


def render_sources_accordion(sources: List[Dict[str, Any]]):
    """Render an expandable accordion detailing source citations, page numbers, and chunk previews."""
    if not sources:
        return

    with st.expander("📚 Sources & Retrieved Context", expanded=False):
        for idx, src in enumerate(sources, start=1):
            file_name = src.get("file_name", "Document")
            page = src.get("page", "N/A")
            score = src.get("score")
            score_text = f" | Distance/Score: `{score}`" if score is not None else ""

            st.markdown(f"**Chunk {idx}: 📄 `{file_name}` — Page {page}**{score_text}")
            st.info(src.get("content", ""))
            st.divider()


def render_sidebar(
    indexed_files: set,
    on_clear_chat: Callable,
    on_clear_db: Callable,
    on_delete_file: Callable,
) -> Dict[str, Any]:
    """Render sidebar controls and return configuration settings."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Provider selection
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["gemini", "groq"],
            index=0,
            help="Select the LLM provider for answering questions.",
        )

        default_key_name = "GOOGLE_API_KEY" if llm_provider == "gemini" else "GROQ_API_KEY"
        preset_key = settings.GOOGLE_API_KEY if llm_provider == "gemini" else settings.GROQ_API_KEY

        # API Key input (masked)
        api_key = st.text_input(
            f"{llm_provider.capitalize()} API Key",
            type="password",
            help=f"Enter key or leave blank if set in .env ({default_key_name})",
        )

        effective_key = api_key or preset_key

        # Model Options
        if llm_provider == "gemini":
            model_options = [
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.6-flash",
                "gemini-flash-latest",
                "gemini-pro-latest",
            ]
            llm_model = st.selectbox(
                "LLM Model",
                options=model_options,
                index=0,
                help="Google Gemini Models (Tested & Verified).",
            )
        else:
            groq_models = list_available_groq_models(effective_key)
            llm_model = st.selectbox(
                "LLM Model",
                options=groq_models,
                index=0,
                help="Active Groq Models (including openai/gpt-oss-120b and openai/gpt-oss-20b).",
            )

        # Allow custom model override
        custom_model = st.text_input(
            "Custom Model ID (Optional)",
            value="",
            help="Type an exact model ID to override the dropdown selection if desired.",
        )
        if custom_model.strip():
            llm_model = custom_model.strip()

        # Embeddings Provider
        embedding_provider = st.selectbox(
            "Embedding Provider",
            options=["gemini", "huggingface"],
            index=0,
            help="Gemini uses verified models/gemini-embedding-001. HuggingFace runs locally on CPU.",
        )

        st.divider()
        st.subheader("🔍 Retrieval & Chunking")

        retrieval_k = st.slider(
            "Top-K Retrieved Chunks (k)",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of most relevant text chunks to retrieve per question.",
        )

        chunk_size = st.slider(
            "Chunk Size",
            min_value=300,
            max_value=2000,
            value=1000,
            step=100,
            help="Maximum characters per chunk.",
        )

        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=150,
            step=25,
            help="Overlap between adjacent chunks to maintain context boundaries.",
        )

        st.divider()
        st.subheader("📁 Indexed Documents")

        if indexed_files:
            for fname in sorted(indexed_files):
                col1, col2 = st.columns([0.8, 0.2])
                col1.write(f"✓ `{fname}`")
                if col2.button("🗑️", key=f"del_{fname}", help=f"Remove {fname}"):
                    on_delete_file(fname)
        else:
            st.info("No documents currently indexed.")

        st.divider()
        st.subheader("🧹 Actions")
        col_c1, col_c2 = st.columns(2)
        if col_c1.button("Clear Chat", use_container_width=True):
            on_clear_chat()
        if col_c2.button("Reset DB", use_container_width=True, type="secondary"):
            on_clear_db()

        return {
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "api_key": api_key,
            "embedding_provider": embedding_provider,
            "retrieval_k": retrieval_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
