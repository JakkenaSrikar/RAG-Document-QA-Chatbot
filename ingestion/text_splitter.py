"""Configurable document chunking service preserving metadata and chunk traceability."""

import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentSplitterService:
    """Splits raw page documents into semantically coherent overlapping chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with unique chunk identifiers added to metadata."""
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)

        # Assign unique chunk_id per document: <file_name>_p<page>_c<index>
        for idx, chunk in enumerate(chunks):
            source = chunk.metadata.get("file_name", "unknown")
            page = chunk.metadata.get("page", 1)
            chunk.metadata["chunk_id"] = f"{source}_p{page}_c{idx}"

        logger.info(
            f"Split {len(documents)} page documents into {len(chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})."
        )
        return chunks
