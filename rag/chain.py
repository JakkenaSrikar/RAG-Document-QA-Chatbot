"""LCEL-based RAG chain with strict anti-hallucination prompt and citation extraction."""

import logging
from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from retrieval.retriever import DocumentRetriever

logger = logging.getLogger(__name__)

STRICT_RAG_SYSTEM_PROMPT = """You are a professional, precise document question-answering assistant.

Answer the user's question using ONLY the information provided in the retrieved context below.

Strict Grounding Rules:
1. Do not assume, extrapolate, or invent information not directly present in the context.
2. Do not use external or outside knowledge.
3. If the answer cannot be found in the provided context, clearly state:
   "I am sorry, but this information is not available in the uploaded documents."
4. Keep your answer factual, direct, and well-structured.
5. Whenever possible, mention which document and page you are referencing.

Retrieved Context:
{context}
"""


class RAGChain:
    """Encapsulates context formatting, prompt templating, and LCEL execution."""

    def __init__(self, llm: BaseChatModel, retriever: DocumentRetriever):
        self.llm = llm
        self.retriever = retriever

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", STRICT_RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        self.output_parser = StrOutputParser()

    @staticmethod
    def format_context(documents: List[Document]) -> str:
        """Format retrieved documents into a clean context string for the prompt."""
        if not documents:
            return "No relevant context found."

        formatted_chunks = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("file_name", "Unknown Document")
            page = doc.metadata.get("page", "?")
            formatted_chunks.append(
                f"--- Document Chunk {i} [Source: {source}, Page: {page}] ---\n{doc.page_content}"
            )
        return "\n\n".join(formatted_chunks)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Execute the end-to-end RAG pipeline for a user question.
        
        Returns:
            Dict with keys: 'answer', 'sources', 'raw_documents'
        """
        # Step 1: Retrieve relevant chunks with scores
        docs_with_scores = self.retriever.get_relevant_documents_with_scores(question)

        if not docs_with_scores:
            return {
                "answer": "No indexed documents found. Please upload and process a PDF document first.",
                "sources": [],
                "raw_documents": [],
            }

        documents = [doc for doc, _ in docs_with_scores]
        context_str = self.format_context(documents)

        # Step 2: Build runnable chain using LCEL
        chain = self.prompt_template | self.llm | self.output_parser

        # Step 3: Generate grounded answer
        try:
            answer = chain.invoke({
                "context": context_str,
                "question": question,
            })
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise RuntimeError(f"Failed to generate answer from language model: {str(e)}")

        # Step 4: Extract structured source metadata
        sources = []
        for doc, score in docs_with_scores:
            sources.append({
                "file_name": doc.metadata.get("file_name", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "score": round(float(score), 4) if score is not None else None,
                "content": doc.page_content,
            })

        return {
            "answer": answer,
            "sources": sources,
            "raw_documents": documents,
        }
