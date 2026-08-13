# Architecture & Engineering Specifications — Unified Enterprise RAG System

## 1. System Overview

The Unified Enterprise RAG System implements a Retrieval-Augmented Generation pipeline using an in-memory vector store powered by NumPy matrix operations.

## 2. Core Modules

### 2.1 Configuration Layer (`config/settings.py`)
- Reads from `st.secrets` when deployed on Streamlit Cloud, falling back to `.env` or `os.environ`.
- Exposes typed methods for API keys, chat models, and embedding models.

### 2.2 Provider Abstraction (`providers/`)
- `BaseAIProvider`: Abstract base class enforcing `embed_documents`, `embed_query`, `generate`, `is_configured`.
- `OpenAIProvider`: Handles OpenAI API calls (`text-embedding-3-small` & `gpt-4o-mini`).
- `GeminiProvider`: Handles Google GenAI SDK calls (`text-embedding-004` & `gemini-2.5-flash`).
- `NvidiaNimProvider`: Handles NVIDIA NIM API calls via OpenAI-compatible SDK (`nv-embedqa-e5-v5` & `nemotron`).

### 2.3 RAG Engine (`rag/`)
- `DocumentLoader`: Extracts clean text from raw strings or PDF files (via `pypdf`).
- `DocumentChunker`: Performs sliding-window splitting with configurable chunk size (default 500) and overlap (default 50).
- `InMemoryVectorStore`: Stores chunk dictionaries paired with vector lists. Enforces vector dimension matching per provider.
- `SemanticRetriever`: Computes exact Cosine Similarity $\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$ using NumPy arrays. Filters by score threshold and sorts descending.
- `RAGPipeline`: Integrates ingestion, retrieval, and system prompt formatting into an end-to-end service.

### 2.4 Security & Logging (`utils/`)
- `security.py`: Implements timing-attack safe access key comparison and regex API key redaction.
- `logging.py`: Provides structured console logging with timestamping.
