"""Retriever module providing similarity search with score ranking and metadata."""

import logging
from typing import List, Tuple
from langchain_core.documents import Document
from vectorstore.chroma_store import ChromaVectorStoreManager

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Executes similarity searches across indexed document chunks."""

    def __init__(self, vector_store_manager: ChromaVectorStoreManager, k: int = 4):
        self.vector_store_manager = vector_store_manager
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve top-k relevant document chunks for a given query."""
        if not query.strip():
            return []

        retriever = self.vector_store_manager.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} chunks for query: '{query[:40]}...'")
        return docs

    def get_relevant_documents_with_scores(
        self, query: str
    ) -> List[Tuple[Document, float]]:
        """Retrieve top-k chunks with associated cosine/L2 similarity scores."""
        if not query.strip():
            return []

        results = self.vector_store_manager.vector_store.similarity_search_with_score(
            query=query, k=self.k
        )
        logger.info(f"Retrieved {len(results)} chunks with scores for query: '{query[:40]}...'")
        return results
