# 📄 RAG-Based Document Q&A Chatbot

A production-ready, interview-grade **Retrieval-Augmented Generation (RAG)** Document Question-Answering application built with **Python, LangChain, ChromaDB, Google Gemini API, Groq API (Llama 3.3 / Mixtral), and Streamlit**.

Users can upload single or multiple PDF documents, automatically index them into a local persistent vector database with semantic chunking and dense embeddings, and ask natural language questions with strictly grounded answers and full page-level source citations.

---

## 🌟 Key Features

- **Multi-Document Ingestion**: Upload and index multiple PDFs simultaneously with SHA256-based duplicate detection.
- **Traceable Text Chunking**: Recursive character text splitting preserving chunk IDs, file provenance, and page numbers.
- **Ultra-Fast LLM Inference with Groq & Gemini**: Choose between **Google Gemini (1.5 Flash/Pro, 2.0 Flash)** and **Groq (Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B)** for lightning-fast answers.
- **Configurable Embeddings**: Seamlessly switch between cloud-based **Google Gemini `models/embedding-001`** and local offline **HuggingFace `sentence-transformers/all-MiniLM-L6-v2`**.
- **Persistent Vector Store (ChromaDB)**: Local vector persistence with support for real-time document deletion and collection reset.
- **Strict Anti-Hallucination Prompting**: System prompt strictly enforces answering exclusively from retrieved document chunks.
- **Transparent Source Citations**: Expandable UI accordions showing exact filename, page number, similarity distance, and chunk text.
- **Session-Based Chat Memory**: Maintains multi-turn conversation context across queries.
- **Modern LangChain LCEL Architecture**: Built on LangChain Expression Language (`ChatPromptTemplate | LLM | StrOutputParser`).

---

## 🏗️ Architecture & Pipeline Flow

```text
                ┌───────────────────────────────┐
                │       PDF Document Upload     │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │   Text & Metadata Extract     │  (PyPDF + Page Numbering)
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │   Recursive Text Chunking     │  (Chunk size: 1000, Overlap: 150)
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │    Embedding Generation       │  (Gemini / HuggingFace)
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │     ChromaDB Vector Store     │  (Persistent Storage & Indexing)
                └───────────────┬───────────────┘
                                │
                                │ (Top-k Similarity Search)
                                ▼
 User Question ──► Query Embedding ──► Retrieved Context Chunks (k=4)
                                             │
                                             ▼
                                  Prompt Template + Context
                                             │
                                             ▼
                                  LLM (Gemini / Groq Llama 3.3)
                                             │
                                             ▼
                               Grounded Answer + Citations
```

---

## 📁 Project Structure

```text
rag-document-qa/
│
├── app.py                      # Main Streamlit Web Application
├── requirements.txt            # Project dependencies with compatible versions
├── .env.example                # Environment variable templates
├── .gitignore                  # Git ignore rules for secrets, venv, and vector data
├── README.md                   # Comprehensive documentation & architecture guide
│
├── config/
│   ├── __init__.py
│   └── settings.py             # Centralized configuration & environment loader
│
├── ingestion/
│   ├── __init__.py
│   ├── pdf_loader.py           # Multi-PDF loader with metadata & error handling
│   └── text_splitter.py        # Configurable recursive chunking engine
│
├── embeddings/
│   ├── __init__.py
│   └── embedding_model.py      # Factory for Google Gemini & HuggingFace embeddings
│
├── vectorstore/
│   ├── __init__.py
│   └── chroma_store.py         # ChromaDB persistence, deduplication, & document deletion
│
├── retrieval/
│   ├── __init__.py
│   └── retriever.py            # Similarity search & metadata-preserving retriever
│
├── generation/
│   ├── __init__.py
│   └── llm.py                  # LLM Factory (Gemini & Groq Models)
│
├── rag/
│   ├── __init__.py
│   └── chain.py                # LCEL RAG pipeline with grounded prompt orchestration
│
├── ui/
│   ├── __init__.py
│   └── components.py           # Streamlit UI layout, sidebar controls, & citations
│
├── data/
│   ├── uploads/                # Temporary PDF upload staging
│   └── chroma/                 # Local vector database persistence directory
│
└── tests/
    ├── __init__.py
    └── test_rag.py             # Unit & integration test suite with mocked APIs
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10 or higher
- Google Gemini API Key and/or Groq API Key

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/JakkenaSrikar/RAG-Document-QA-Chatbot.git
cd RAG-Document-QA-Chatbot

# Create and activate virtual environment
# On Windows:
python -m venv venv
venv\Scripts\activate

# On Linux/macOS:
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```

Edit `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
EMBEDDING_PROVIDER=gemini
```

### 5. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Run the test suite with pytest:
```bash
pytest tests/ -v
```

---

## 💡 Key Technical Explanations (For Technical Interviews)

### 1. Why is Chunking Necessary?
LLMs have finite context windows, and passing entire 200-page PDFs directly causes token exhaustion, high inference latency, and high cost. Moreover, attention mechanisms suffer from the **"Lost in the Middle"** phenomenon. Chunking breaks large text into discrete, semantically focused units with overlap (e.g. 150 characters) to ensure context is never split across arbitrary boundaries.

### 2. How Do Vector Embeddings Work?
Embedding models map high-dimensional text into dense mathematical vectors (e.g. 768 dimensions for Gemini). Semantic similarity corresponds to geometric proximity in vector space, measured using **Cosine Similarity** or **Euclidean Distance (L2)**.

### 3. Why ChromaDB?
ChromaDB is a purpose-built open-source vector store that enables local persistence, embedded in-process execution (eliminating heavy external database servers for prototyping), metadata filtering, and fast Approximate Nearest Neighbor (ANN) index searches using HNSW (Hierarchical Navigable Small World graphs).

### 4. Why Groq for LLM Inference?
Groq's LPU (Language Processing Unit) architecture delivers ultra-low latency token generation (up to 500+ tokens/second) for open-weight models like Llama 3.3 70B and Mixtral 8x7B, making RAG chatbots feel instantaneous compared to standard cloud API endpoints.

### 5. How Does This System Prevent Hallucinations?
1. **Explicit Grounding Instruction**: The system prompt strictly prohibits the LLM from using outside pre-training knowledge.
2. **Explicit Refusal Instruction**: If the answer is absent from the retrieved chunks, the model is instructed to output a standardized refusal rather than guessing.
3. **Traceable Citations**: Every answer is paired with the exact chunks and page numbers retrieved, enabling instant human verification.

---

## 💼 Resume Bullet Points

- **RAG-Based Document Q&A Chatbot | Python, LangChain, ChromaDB, Google Gemini, Groq, Streamlit**
  - Engineered an end-to-end Retrieval-Augmented Generation (RAG) web app enabling semantic question-answering over multi-document PDF uploads with sub-second inference using Groq and Gemini.
  - Implemented an ingestion pipeline using PyPDF, recursive character chunking, and ChromaDB vector persistence with SHA256-based deduplication and metadata tracking.
  - Formulated strict anti-hallucination prompt templates with LangChain LCEL to ensure responses remain 100% grounded in retrieved contexts, complete with page-level citations.
