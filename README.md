# Unified Enterprise RAG System

A modular, production-ready Retrieval-Augmented Generation (RAG) platform built with Python, NumPy, and Streamlit. 

It provides unified AI provider abstraction across **OpenAI**, **Google Gemini**, and **NVIDIA NIM**, enabling document ingestion, sliding-window text chunking, in-memory vector indexing, exact cosine similarity retrieval, and citation-backed question answering.

---

## 🌟 Key Features

* **Unified AI Provider Abstraction**: A single selected provider (OpenAI, Google Gemini, or NVIDIA NIM) owns both vector embedding generation and chat generation.
* **In-Memory Vector Search Engine**: Stores text chunks and high-dimensional embeddings directly in memory with NumPy matrix operations for exact Cosine Similarity evaluation.
* **Vector Space Dimension Protection**: Automatically validates embedding dimensions (e.g. `1536` for OpenAI, `768` for Gemini) to prevent cross-provider vector distance calculation errors.
* **Sliding Window Chunker**: Configurable chunk size and overlap parameters preserving source metadata across text splits.
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
   Active Provider ──► Generate vector embeddings (e.g., 1536-dim)
        │
        ▼
InMemoryVectorStore ──► Store chunk text + embeddings in memory
        │
        ▼
SemanticRetriever ──► Compute Cosine Similarity via NumPy (Top-K)
        │
        ▼
   Active Provider ──► Generate citation-backed RAG completion
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and set your preferred provider API keys:

```bash
cp .env.example .env
```

Example `.env`:
```ini
APP_ACCESS_KEY=admin123
AI_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Running the Web Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`, enter the access key (`admin123`), upload a PDF or TXT file, index documents, and begin asking questions!

---

## 🧪 Running Unit Tests

Run the test suite using `pytest` or Python's built-in `unittest`:

```bash
pytest tests/
```

Or:

```bash
python3 -m unittest discover tests/
```

---

## 🔒 Security Best Practices

* Secrets are loaded dynamically via `process.env` / `st.secrets`—never committed to source code.
* Constant-time string comparison (`hmac.compare_digest`) prevents timing side-channel attacks on access gates.
* Errors are sanitized using regex redaction before logging or rendering in the UI.
