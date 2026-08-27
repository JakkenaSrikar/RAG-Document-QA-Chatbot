"""ChromaDB integration with deduplication, local persistence, and document lifecycle management."""

import logging
import os
import shutil
from typing import List, Set, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from config.settings import settings

logger = logging.getLogger(__name__)


class ChromaVectorStoreManager:
    """Manages persistent ChromaDB operations: indexing, deduplication, search, and deletion."""

    def __init__(
        self,
        embedding_function: Embeddings,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or settings.COLLECTION_NAME

        os.makedirs(self.persist_directory, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )

    def get_indexed_files(self) -> Set[str]:
        """Retrieve the list of unique filenames currently indexed in the vector store."""
        try:
            collection_data = self.vector_store.get()
            metadatas = collection_data.get("metadatas", [])
            indexed_files = {m["file_name"] for m in metadatas if m and "file_name" in m}
            return indexed_files
        except Exception as e:
            logger.warning(f"Could not retrieve indexed files: {e}")
            return set()

    def is_document_indexed(self, filename: str, file_hash: Optional[str] = None) -> bool:
        """Check if a document with the same filename or hash has already been stored."""
        try:
            collection_data = self.vector_store.get()
            metadatas = collection_data.get("metadatas", [])
            for m in metadatas:
                if not m:
                    continue
                if file_hash and m.get("file_hash") == file_hash:
                    return True
                if m.get("file_name") == filename:
                    return True
            return False
        except Exception:
            return False

    def add_documents(self, chunks: List[Document]) -> int:
        """Index a list of chunked documents into ChromaDB.
        
        Returns:
            Number of chunks successfully indexed.
        """
        if not chunks:
            return 0

        # Extract IDs based on metadata chunk_id
        ids = [chunk.metadata.get("chunk_id") for chunk in chunks]
        self.vector_store.add_documents(documents=chunks, ids=ids)
        logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB collection '{self.collection_name}'.")
        return len(chunks)

    def delete_document(self, filename: str) -> bool:
        """Delete all chunks belonging to a specific document from ChromaDB."""
        try:
            collection = self.vector_store._collection
            collection.delete(where={"file_name": filename})
            logger.info(f"Deleted all vectors for file: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document '{filename}': {e}")
            return False

    def clear_all(self):
        """Completely clear and reset the vector database."""
        try:
            self.vector_store.delete_collection()
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                os.makedirs(self.persist_directory, exist_ok=True)
            # Reinitialize empty collection
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory,
            )
            logger.info("ChromaDB vector store cleared successfully.")
        except Exception as e:
            logger.error(f"Error while clearing ChromaDB: {e}")
            raise
