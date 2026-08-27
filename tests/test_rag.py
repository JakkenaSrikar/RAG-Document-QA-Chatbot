"""Unit and integration tests for RAG pipeline components."""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeChatModel

from ingestion.text_splitter import DocumentSplitterService
from rag.chain import RAGChain, STRICT_RAG_SYSTEM_PROMPT


class TestDocumentSplitter:
    """Test text splitting and chunk metadata tagging."""

    def test_split_documents_preserves_metadata(self):
        splitter = DocumentSplitterService(chunk_size=100, chunk_overlap=20)
        docs = [
            Document(
                page_content="Artificial Intelligence and Machine Learning are transformative technologies. " * 5,
                metadata={"file_name": "ai_report.pdf", "page": 1},
            )
        ]
        chunks = splitter.split_documents(docs)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["file_name"] == "ai_report.pdf"
            assert chunk.metadata["page"] == 1
            assert "chunk_id" in chunk.metadata
            assert chunk.metadata["chunk_id"].startswith("ai_report.pdf_p1_c")

    def test_split_empty_documents_returns_empty(self):
        splitter = DocumentSplitterService()
        assert splitter.split_documents([]) == []


class TestRAGChain:
    """Test RAG chain context construction, grounding, and output generation."""

    def test_format_context_with_documents(self):
        docs = [
            Document(
                page_content="Revenue grew by 24% year-over-year.",
                metadata={"file_name": "financials.pdf", "page": 3},
            ),
            Document(
                page_content="Operating expenses remained flat.",
                metadata={"file_name": "financials.pdf", "page": 4},
            ),
        ]
        context = RAGChain.format_context(docs)

        assert "--- Document Chunk 1 [Source: financials.pdf, Page: 3] ---" in context
        assert "Revenue grew by 24% year-over-year." in context
        assert "--- Document Chunk 2 [Source: financials.pdf, Page: 4] ---" in context

    def test_format_context_empty(self):
        context = RAGChain.format_context([])
        assert context == "No relevant context found."

    def test_rag_chain_execution_with_fake_llm(self):
        mock_retriever = MagicMock()
        mock_doc = Document(
            page_content="The solar panel efficiency is 22.5%.",
            metadata={"file_name": "solar.pdf", "page": 2, "chunk_id": "solar_p2_c0"},
        )
        mock_retriever.get_relevant_documents_with_scores.return_value = [(mock_doc, 0.15)]

        fake_llm = FakeChatModel(responses=["The solar panel efficiency is 22.5% according to solar.pdf (Page 2)."])
        rag_chain = RAGChain(llm=fake_llm, retriever=mock_retriever)

        result = rag_chain.answer_question("What is the efficiency?")

        assert "efficiency is 22.5%" in result["answer"]
        assert len(result["sources"]) == 1
        assert result["sources"][0]["file_name"] == "solar.pdf"
        assert result["sources"][0]["page"] == 2
        assert result["sources"][0]["score"] == 0.15

    def test_rag_chain_no_indexed_documents(self):
        mock_retriever = MagicMock()
        mock_retriever.get_relevant_documents_with_scores.return_value = []

        fake_llm = FakeChatModel(responses=[""])
        rag_chain = RAGChain(llm=fake_llm, retriever=mock_retriever)

        result = rag_chain.answer_question("What is the revenue?")
        assert "No indexed documents found" in result["answer"]
        assert result["sources"] == []
