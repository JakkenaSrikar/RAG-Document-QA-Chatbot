"""Reusable UI helper components for Streamlit interface."""

from typing import List, Dict, Any, Callable
import streamlit as st


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
            options=["gemini", "openai"],
            index=0,
            help="Select the LLM provider for answering questions.",
        )

        # Dynamic Model Options
        if llm_provider == "gemini":
            model_options = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]
            default_key_name = "GOOGLE_API_KEY"
        else:
            model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
            default_key_name = "OPENAI_API_KEY"

        llm_model = st.selectbox("LLM Model", options=model_options, index=0)

        # API Key input (masked)
        api_key = st.text_input(
            f"{llm_provider.capitalize()} API Key",
            type="password",
            help=f"Enter key or leave blank if set in .env ({default_key_name})",
        )

        # Embeddings Provider
        embedding_provider = st.selectbox(
            "Embedding Provider",
            options=["gemini", "huggingface"],
            index=0,
            help="Gemini embeddings require an API key. HuggingFace runs locally on CPU.",
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
