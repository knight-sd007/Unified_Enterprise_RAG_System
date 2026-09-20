# Architecture & Engineering Specifications — Unified Enterprise RAG System

## 1. System Overview

The Unified Enterprise RAG System implements a production-grade Retrieval-Augmented Generation pipeline supporting:
- Cloud-scale vector search via **Qdrant Cloud** with deterministic collection routing.
- Fast in-memory vector indexing with NumPy matrix cosine similarity for standalone execution.
- Strict provider identity and vector-space isolation.
- Deterministic duplicate-ingestion protection via UUIDv5 identifiers.
- Prompt trust-boundary hardening treating retrieved context as untrusted passive reference data.

## 2. Core Modules

### 2.1 Configuration Layer (`config/settings.py`)
- Reads dynamically from `st.secrets` when deployed on Streamlit Cloud, falling back to `.env` or `os.environ`.
- Exposes typed methods for AI provider keys, chat/embedding models, and Qdrant credentials (`QDRANT_URL`, `QDRANT_API_KEY`).
- Centralizes deterministic provider vector space specifications:
  - **Google Gemini**: 768 dimensions, collection `p06_gemini_text_embedding_004`, Cosine distance.
  - **NVIDIA NIM**: 1024 dimensions, collection `p06_nvidia_nv_embedqa_e5_v5`, Cosine distance.
  - **OpenAI**: 1536 dimensions, collection `p06_openai_text_embedding_3_small`, Cosine distance (when provisioned).

### 2.2 Provider Abstraction (`providers/`)
- `BaseAIProvider`: Abstract base class enforcing `embed_documents`, `embed_query`, `generate`, `is_configured`.
- `OpenAIProvider`: Handles OpenAI API calls (`text-embedding-3-small` & `gpt-4o-mini`).
- `GeminiProvider`: Handles Google GenAI SDK calls (`text-embedding-004` & `gemini-2.5-flash`) with strict vector validation (rejecting empty/non-numeric/malformed vectors).
- `NvidiaNimProvider`: Handles NVIDIA NIM API calls via OpenAI-compatible SDK (`nv-embedqa-e5-v5` & `nemotron`).

### 2.3 RAG Engine (`rag/`)
- `DocumentLoader`: Extracts clean text from raw strings or PDF files (via `pypdf`).
- `DocumentChunker`: Performs sliding-window splitting with configurable chunk size and overlap.
- `QdrantVectorStore`: Manages upserts and vector queries on Qdrant Cloud. Enforces collection existence, provider matching, and deterministic UUIDv5 point IDs for idempotent deduplication.
- `InMemoryVectorStore`: Manages local in-memory storage with exact NumPy cosine similarity and provider isolation.
- `SemanticRetriever`: Computes similarity against active vector store with threshold filtering and top-K ranking.
- `RAGPipeline`: Integrates ingestion, retrieval, and prompt formatting into an end-to-end service.
- `prompt.py`: Formats user questions and retrieved snippets, explicitly enforcing the **untrusted passive reference data** boundary in system instructions.

### 2.4 Security & Logging (`utils/`)
- `security.py`: Implements constant-time access key verification (`hmac.compare_digest`) and regex-based secret redaction (API keys, tokens, credentials in URLs).
- `logging.py`: Structured console logger with timestamping and severity levels.
