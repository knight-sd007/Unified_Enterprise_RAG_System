# Unified Enterprise RAG System

A modular, production-ready Retrieval-Augmented Generation (RAG) platform built with Python, Qdrant Cloud, NumPy, and Streamlit.

It provides unified AI provider abstraction across **Google Gemini**, **NVIDIA NIM**, and **OpenAI**, enabling document ingestion, sliding-window text chunking, persistent Qdrant Cloud and in-memory vector indexing, deterministic duplicate protection, prompt trust-boundary hardening, and citation-backed question answering.

---

## 🌟 Key Features

* **Unified AI Provider Abstraction**: A single selected provider (OpenAI, Google Gemini, or NVIDIA NIM) owns both vector embedding generation and chat generation.
* **Qdrant Cloud Vector Database**: Persistent vector search engine with provider-specific collection routing:
  * **Google Gemini**: `p06_gemini_text_embedding_004` (768 dimensions, Cosine)
  * **NVIDIA NIM**: `p06_nvidia_nv_embedqa_e5_v5` (1024 dimensions, Cosine)
  * **OpenAI**: `p06_openai_text_embedding_3_small` (1536 dimensions, Cosine, when provisioned)
* **In-Memory Fallback Engine**: Local standalone vector search with NumPy matrix operations for exact Cosine Similarity evaluation.
* **Vector Space Isolation & Dimension Protection**: Enforces provider identity and strict dimension checking (Gemini `768`, NVIDIA `1024`, OpenAI `1536`) to prevent cross-provider vector contamination.
* **Deterministic Duplicate-Ingestion Protection**: Generates deterministic UUIDv5 identifiers from chunk content and metadata for idempotent re-indexing.
* **Prompt Trust Boundary**: Treats all retrieved context snippets as untrusted passive reference data to mitigate prompt injection.
* **Source Citation Back-References**: Answers include explicit file and chunk ID citations with similarity relevance scores.
* **Authentication Access Gate**: Constant-time comparison access key protection (`APP_ACCESS_KEY`).
* **Security & Secret Sanitization**: Redacts API keys and sensitive credentials from error logs and UI outputs.

---

## 🏗️ Architecture Overview

```text
Uploaded Files (PDF / TXT)
        │
        ▼
   DocumentLoader ──► Extract raw text & metadata
        │
        ▼
  DocumentChunker ──► Split into overlapping chunks
        │
        ▼
   Active Provider ──► Generate validated embeddings (e.g., 768-dim / 1024-dim)
        │
        ▼
QdrantVectorStore ──► Store chunk text + vectors in provider collection (UUIDv5)
        │
        ▼
SemanticRetriever ──► Query top-K chunks with similarity threshold filtering
        │
        ▼
   Active Provider ──► Generate citation-backed RAG answer (Untrusted Context)
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

Example `.env`:
```ini
APP_ACCESS_KEY=admin123
AI_PROVIDER=gemini

# Provider Credentials
GEMINI_API_KEY=your_gemini_api_key_here
NVIDIA_API_KEY=your_nvidia_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Cloud (Optional for cloud vector storage)
QDRANT_URL=https://your-cluster.qdrant.tech:6333
QDRANT_API_KEY=your_qdrant_api_key_here
```

### 3. Running the Web Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`, enter the access key (`admin123`), upload a PDF or TXT file, index documents, and begin asking questions!

---

## 🧪 Running Unit Tests

Run the test suite using `pytest`:

```bash
pytest tests/
```

Or:

```bash
python3 -m unittest discover tests/
```

## 🔒 Security Best Practices

* Secrets are loaded dynamically via `os.environ` / `st.secrets`—never committed to source code.
* Constant-time string comparison (`hmac.compare_digest`) prevents timing side-channel attacks on access gates.
* Errors and logs are sanitized using regex redaction before output.
* Retrieved document content is marked as untrusted passive reference data.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
