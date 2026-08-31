"""Production-Ready Streamlit RAG Application.
Demonstrates: PDF Ingestion -> Text Splitter -> Vector Embeddings -> ChromaDB -> LCEL RAG Chain -> Grounded UI
"""

import logging
import streamlit as st

from config.settings import settings
from ingestion.pdf_loader import PDFLoaderService
from ingestion.text_splitter import DocumentSplitterService
from embeddings.embedding_model import get_embedding_model
from vectorstore.chroma_store import ChromaVectorStoreManager
from retrieval.retriever import DocumentRetriever
from generation.llm import get_llm
from rag.chain import RAGChain
from ui.components import render_sidebar, render_sources_accordion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit Page Setup
st.set_page_config(
    page_title="RAG Document Q&A Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Cached resource factories keyed strictly by provider and collection
@st.cache_resource(show_spinner=False)
def load_embedding_model(provider: str, api_key: str):
    """Cache embedding model instance to avoid expensive reloads."""
    return get_embedding_model(provider=provider, api_key=api_key)


@st.cache_resource(show_spinner=False)
def load_vector_store(_embedding_function, collection_name: str, persist_dir: str):
    """Cache vector database connection manager isolated per collection name."""
    return ChromaVectorStoreManager(
        embedding_function=_embedding_function,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


def handle_clear_chat():
    st.session_state.messages = []
    st.rerun()


def handle_clear_db(vector_manager: ChromaVectorStoreManager):
    vector_manager.clear_all()
    st.session_state.processed_files = set()
    st.session_state.messages = []
    st.cache_resource.clear()
    st.success("Vector database and cache reset successfully!")
    st.rerun()


def handle_delete_file(vector_manager: ChromaVectorStoreManager, filename: str):
    success = vector_manager.delete_document(filename)
    if success:
        st.session_state.processed_files.discard(filename)
        st.success(f"Removed '{filename}' from database.")
        st.rerun()
    else:
        st.error(f"Failed to delete '{filename}'.")


def main():
    st.title("📄 RAG Document Q&A Assistant")
    st.caption("Upload documents and ask questions grounded strictly in their content.")

    # 1. Render Sidebar first to read user's chosen provider
    indexed_files = set()

    # Pre-render sidebar
    config = render_sidebar(
        indexed_files=indexed_files,
        on_clear_chat=handle_clear_chat,
        on_clear_db=lambda: handle_clear_db(vector_manager) if 'vector_manager' in locals() else None,
        on_delete_file=lambda fname: handle_delete_file(vector_manager, fname) if 'vector_manager' in locals() else None,
    )

    api_key_to_use = config["api_key"] or (
        settings.GOOGLE_API_KEY if config["llm_provider"] == "gemini" else settings.GROQ_API_KEY
    )

    # Isolated collection per embedding provider (prevents dimension mismatch between 384 vs 3072)
    provider_collection_name = f"{settings.COLLECTION_NAME}_{config['embedding_provider']}"

    # Initialize current embedding & store manager strictly based on chosen provider
    try:
        embed_key = config["api_key"] if config["embedding_provider"] == "gemini" else settings.GOOGLE_API_KEY
        embedding_model = load_embedding_model(
            provider=config["embedding_provider"],
            api_key=embed_key,
        )
        vector_manager = load_vector_store(
            _embedding_function=embedding_model,
            collection_name=provider_collection_name,
            persist_dir=settings.CHROMA_PERSIST_DIRECTORY,
        )
        indexed_files = vector_manager.get_indexed_files()
    except Exception as e:
        st.error(f"⚠️ Embedding Model Initialization Error: {str(e)}")
        st.stop()

    # 2. File Upload & Ingestion Section
    st.subheader("1. Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select PDF documents to index into the vector store.",
    )

    if uploaded_files:
        process_btn = st.button("🚀 Process and Index Documents", type="primary")
        if process_btn:
            total_chunks_added = 0
            with st.spinner("Processing documents, extracting text, and generating embeddings..."):
                splitter = DocumentSplitterService(
                    chunk_size=config["chunk_size"],
                    chunk_overlap=config["chunk_overlap"],
                )

                for file in uploaded_files:
                    try:
                        # Extract pages
                        docs, filename, file_hash = PDFLoaderService.save_and_extract(
                            uploaded_file=file,
                            upload_dir=settings.UPLOAD_DIRECTORY,
                        )

                        # Check deduplication
                        if vector_manager.is_document_indexed(filename, file_hash):
                            st.warning(f"ℹ️ '{filename}' is already indexed. Skipping.")
                            continue

                        # Chunk document
                        chunks = splitter.split_documents(docs)

                        # Store in ChromaDB
                        added = vector_manager.add_documents(chunks)
                        total_chunks_added += added
                        st.session_state.processed_files.add(filename)
                        st.success(f"✓ Indexed **{filename}** ({len(docs)} pages, {added} chunks)")

                    except Exception as e:
                        logger.error(f"Error processing {file.name}: {e}")
                        st.error(f"⚠️ Unable to process '{file.name}': {str(e)}")

            if total_chunks_added > 0:
                st.balloons()
                st.rerun()

    st.divider()

    # 3. Chat Interface
    st.subheader("2. Ask Questions Grounded in Documents")

    # Display past conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                render_sources_accordion(message["sources"])

    # Handle user query
    user_query = st.chat_input("Ask a question about your uploaded documents...")
    if user_query:
        # Check for API key
        if not api_key_to_use:
            st.error(
                f"⚠️ Please enter your {config['llm_provider'].capitalize()} API Key in the sidebar "
                "or set it in your `.env` file to generate answers."
            )
            st.stop()

        # Check if database has files
        current_indexed = vector_manager.get_indexed_files()
        if not current_indexed:
            st.warning(f"⚠️ No documents indexed yet with '{config['embedding_provider']}'. Please upload and process a PDF above.")
            st.stop()

        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Searching document context and generating answer..."):
                try:
                    # Initialize LLM
                    llm = get_llm(
                        provider=config["llm_provider"],
                        model_name=config["llm_model"],
                        api_key=api_key_to_use,
                        temperature=0.0,  # Deterministic grounding
                    )

                    # Initialize Retriever
                    retriever = DocumentRetriever(
                        vector_store_manager=vector_manager,
                        k=config["retrieval_k"],
                    )

                    # Run RAG Chain
                    rag_chain = RAGChain(llm=llm, retriever=retriever)
                    response = rag_chain.answer_question(user_query)

                    answer_text = response["answer"]
                    sources = response["sources"]

                    st.markdown(answer_text)
                    if sources:
                        render_sources_accordion(sources)

                    # Store in chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                    })

                except Exception as e:
                    logger.error(f"Error answering question: {e}")
                    error_msg = f"⚠️ An error occurred while generating the response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                    })


if __name__ == "__main__":
    main()
