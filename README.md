# Unified Enterprise RAG System

A modular, production-ready Retrieval-Augmented Generation (RAG) platform built with Python, FastAPI, React, TypeScript, Tailwind CSS, Qdrant Cloud, and NumPy.

It provides unified AI provider abstraction across **Google Gemini**, **NVIDIA NIM**, and **OpenAI**, enabling document ingestion, sliding-window text chunking, persistent Qdrant Cloud and in-memory vector indexing, deterministic duplicate protection, prompt trust-boundary hardening, and citation-backed question answering.

---

## 🌟 Key Features

* **Unified AI Provider Abstraction**: A single selected provider (OpenAI, Google Gemini, or NVIDIA NIM) owns both vector embedding generation and chat generation.
* **Qdrant Cloud Vector Database**: Persistent vector search engine with provider-specific collection routing:
  * **Google Gemini**: `p06_gemini_embedding_2_768` (768 dimensions, Cosine)
  * **NVIDIA NIM**: `p06_nvidia_llama_nemotron_embed_1b_v2_2048` (2048 dimensions, Cosine)
  * **OpenAI**: `p06_openai_text_embedding_3_small` (1536 dimensions, Cosine, when provisioned)
* **In-Memory Fallback Engine**: Local standalone vector search with NumPy matrix operations for exact Cosine Similarity evaluation.
* **Vector Space Isolation & Dimension Protection**: Enforces provider identity and strict dimension checking (Gemini `768`, NVIDIA `2048`, OpenAI `1536`) to prevent cross-provider vector contamination.
* **Deterministic Duplicate-Ingestion Protection**: Generates deterministic UUIDv5 identifiers from chunk content and metadata for idempotent re-indexing.
* **Prompt Trust Boundary**: Treats all retrieved context snippets as untrusted passive reference data to mitigate prompt injection.
* **Source Citation Back-References**: Answers include explicit file and chunk ID citations with similarity relevance scores.
* **FastAPI Authoritative REST API**: Production-grade REST backend exposing `/api/v1/health`, `/api/v1/auth/*`, `/api/v1/providers`, `/api/v1/documents/*`, and `/api/v1/rag/*`.
* **React + TypeScript + Vite Frontend**: Dual-pane enterprise workspace with live telemetry HUD, drag-and-drop document ingestion, grounded Q&A with expandable citations, and diagnostics.
* **Single-Origin Deployment**: Frontend static assets served directly from FastAPI or via reverse-proxy on `https://rag.vaikuntrix.in/`.
* **Security & Secret Sanitization**: Redacts API keys and sensitive credentials from error logs and UI outputs. Constant-time signature verification on auth tokens.

---

## 🏗️ Architecture Overview

```text
Browser / Client (React 18 + TypeScript + Vite SPA)
                    │
                    ▼  (HTTPS / REST)
       FastAPI Application Entrypoint (:8000)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  /api/v1/documents/ingest   /api/v1/rag/query
        │                       │
        ▼                       ▼
  DocumentLoader         SemanticRetriever
  DocumentChunker               │
        │                       ▼
        ▼                Active AI Provider (Gemini / NVIDIA / OpenAI)
 Qdrant Cloud / NumPy           │
  (UUIDv5 Deduplicated)         ▼
                         Citation-Backed Response
```

---

## 🚀 Quickstart Guide

### 1. Backend Setup & Installation

Clone the repository and install backend Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Frontend Setup & Build

Install frontend dependencies and compile static assets:

```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and configure your credentials:

```bash
cp .env.example .env
```

Example `.env`:
```ini
APP_ACCESS_KEY=your-secure-access-key
AI_PROVIDER=gemini

# Provider Credentials
GEMINI_API_KEY=your_gemini_api_key_here
NVIDIA_API_KEY=your_nvidia_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Cloud (Optional for cloud vector storage)
QDRANT_URL=https://your-cluster.qdrant.tech:6333
QDRANT_API_KEY=your_qdrant_api_key_here
```

### 4. Running the Application

Launch the FastAPI unified service (serves both API and built React SPA):

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open your browser at `http://localhost:8000`, enter your configured `APP_ACCESS_KEY`, upload documents, and explore multi-provider grounded RAG!

For frontend development with hot-reloading:

```bash
cd frontend
npm run dev
```

---

## 🧪 Running Automated Tests

Run the complete test suite using `pytest`:

```bash
pytest tests/
```

Or:

```bash
python3 -m unittest discover tests/
```

---

## 🔒 Security Best Practices

* **No Hardcoded Secrets**: Secrets are loaded exclusively from runtime environment variables (`os.environ`).
* **Constant-Time Verification**: Uses `hmac.compare_digest` for access key verification and token signatures.
* **Redaction & Sanitization**: Sensitive API keys and internal stack traces are redacted before client responses.
* **Least Privilege Container**: Runs as unprivileged non-root user (`UID: 65532`) on Distroless Linux base.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
