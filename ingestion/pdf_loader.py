"""Robust PDF ingestion service supporting single/multiple PDFs and error recovery."""

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFLoaderService:
    """Handles PDF validation, saving, text extraction, and metadata enrichment."""

    @staticmethod
    def calculate_file_hash(file_bytes: bytes) -> str:
        """Compute SHA256 hash to uniquely identify documents and avoid duplicate indexing."""
        return hashlib.sha256(file_bytes).hexdigest()

    @classmethod
    def save_and_extract(
        cls, uploaded_file, upload_dir: str
    ) -> Tuple[List[Document], str, str]:
        """Save an uploaded file from Streamlit, validate it, and extract Document objects with metadata.
        
        Returns:
            Tuple of (documents, filename, file_hash)
        """
        filename = uploaded_file.name
        if not filename.lower().endswith(".pdf"):
            raise ValueError(f"Unsupported file format for '{filename}'. Only PDF files are supported.")

        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) == 0:
            raise ValueError(f"Uploaded file '{filename}' is empty (0 bytes).")

        file_hash = cls.calculate_file_hash(file_bytes)
        file_path = Path(upload_dir) / filename

        # Save to disk
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        logger.info(f"Saved uploaded PDF: {filename} ({len(file_bytes)} bytes)")

        # Extract text via PyPDF
        try:
            loader = PyPDFLoader(str(file_path))
            raw_docs = loader.load()
        except Exception as e:
            logger.error(f"PyPDFLoader failed on {filename}: {e}")
            raise RuntimeError(f"Corrupted or unreadable PDF '{filename}': {str(e)}")

        if not raw_docs:
            raise ValueError(f"No pages could be extracted from '{filename}'.")

        # Validate that document contains actual readable text (not empty or purely scanned without OCR)
        total_text_length = sum(len(doc.page_content.strip()) for doc in raw_docs)
        if total_text_length < 20:
            raise ValueError(
                f"PDF '{filename}' contains little or no extractable text. "
                "It may be a scanned image-only PDF."
            )

        # Standardize and enrich metadata
        enriched_docs = []
        for doc in raw_docs:
            page_num = doc.metadata.get("page", 0) + 1  # 1-indexed page number
            clean_content = doc.page_content.strip()
            if clean_content:
                enriched_doc = Document(
                    page_content=clean_content,
                    metadata={
                        "source": filename,
                        "file_name": filename,
                        "page": page_num,
                        "file_hash": file_hash,
                        "source_path": str(file_path),
                    },
                )
                enriched_docs.append(enriched_doc)

        logger.info(
            f"Successfully extracted {len(enriched_docs)} pages from {filename}."
        )
        return enriched_docs, filename, file_hash
