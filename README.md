# Unified Enterprise RAG System

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-Jenkins%20Declarative-blue.svg)](Jenkinsfile)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%7C%20Python%203.12-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%7C%20TypeScript%20%7C%20Vite-61DAFB.svg)](frontend/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant%20Cloud%20%7C%20NumPy-DC2626.svg)](https://qdrant.tech)
[![Security](https://img.shields.io/badge/Security-Distroless%20Nonroot%20%7C%20Gitleaks%20%7C%20Trivy-4CAF50.svg)](#-security--hardening)
[![License](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

A production-grade, multi-provider Retrieval-Augmented Generation (RAG) platform engineered with Python 3.12, FastAPI, React 18, TypeScript, Tailwind CSS, Qdrant Cloud, SQLite Metadata Persistence, Google OAuth & Drive storage, and NumPy.

The platform unifies vector embeddings, semantic retrieval, prompt trust-boundary hardening, and grounded question answering across **Google Gemini**, **NVIDIA NIM**, and **OpenAI** under an isolated vector space architecture.

---

## 🏗️ System Architecture

```text
                                  Browser Client
                   (React 18 + TypeScript + Tailwind CSS SPA)
                                       │
                                       ▼  (HTTPS / REST)
                   Cloudflare Edge / Tunnel Ingress
                       (https://rag.vaikuntrix.in)
                                       │
                                       ▼
                       FastAPI Application Gateway
                       (:8000 Container / :8006 Host)
                                       │
         ┌─────────────────────────────┼─────────────────────────────┬─────────────────────────────┐
         ▼                             ▼                             ▼                             ▼
   /api/v1/auth/*              /api/v1/documents/*              /api/v1/rag/*                /api/v1/admin/*
 (Access Key Gate,             (Upload, Ingestion,          (Retrieval & Decoupled       (Privileged Overview,
  OAuth & Drive Auth)           SQLite & Drive Sync)          QA Execution Pipeline)      Diagnostics & Purge)
         │                             │                             │                             │
         ▼                             ▼                             ▼                             ▼
   Google OAuth                  DocumentLoader              SemanticRetriever            Metadata Repository
   & Fernet Token               DocumentChunker                      │                    (SQLite data/p06.db)
     Encryption              (Sliding-Window / UUIDv5)               ▼                             │
         │                             │                    Active AI Provider                     ▼
         ▼                             ▼                 (Gemini / NVIDIA / OpenAI)       Live Health Probes
   Google Drive              Vector Store Dispatcher                 │                   (/health/providers)
   (drive.file)              ┌─────────────┴─────────────┐           ▼
   Original Sync             ▼                           ▼    Grounded Answer +
                       Qdrant Cloud             In-Memory Store  Citations & Scores
                 (Provider Collections)             (NumPy)
```

### Frontend & Backend Unified Serving

The FastAPI backend operates as a single-origin application server:
1. **API Endpoints**: Routed under `/api/v1/*`.
2. **Interactive Documentation**: Served under `/docs` (Swagger UI), `/redoc` (ReDoc), and `/openapi.json` (OpenAPI 3.1 Specification).
3. **Frontend Static Assets**: Compiled React assets located in `frontend/dist/` are served directly by FastAPI. Static JavaScript and CSS bundles are mounted under `/assets`, and client-side routes fallback to `frontend/dist/index.html` without intercepting `/api/*`, `/docs`, `/redoc`, or `/openapi.json`.

---

## 🌟 Core Features

- **Multi-User Document & Vector Isolation**: Complete tenant/user isolation across ingestion, retrieval, listing, and deletion. Each document and vector point is bound to a verified server-side `user_id`. Queries strictly search within the authenticated user's vector space.
- **Durable SQLite Metadata Repository**: Stores document records, chunk distributions, file sizes, character counts, and encrypted credentials in `data/p06_metadata.db`.
- **Google OAuth / OpenID Connect & Drive Integration**:
  - Authenticates users with Google OAuth using verified `google_<sub_id>` identifiers.
  - Server-side admin email/sub allowlist (`GOOGLE_ADMIN_EMAILS`) for automated privilege escalation.
  - Optional user-consented source file backup to personal Google Drive with `https://www.googleapis.com/auth/drive.file` scope.
  - Fernet-encrypted OAuth token storage at rest.
- **Decoupled & Dynamic Model Execution**:
  - Independent selection of text-generation model (`chat_provider_id` + `chat_model`) and embedding model (`embedding_provider_id` + `embedding_model`).
  - Allows semantic vector retrieval against Gemini/NVIDIA vector spaces while synthesizing final grounded responses via OpenAI GPT-4o.
- **Accurate Provider Health & Connectivity Diagnostics**:
  - Dedicated `/health/providers` endpoint distinguishes between overall app health (`healthy`), unconfigured provider credentials (`not_configured`), and live API connectivity verification (`connected` / `unreachable`).
- **Secure Administrator Console**:
  - Dedicated `/api/v1/admin/*` routes providing aggregate tenant metrics, live connectivity audits with latency measurement, and two-step global vector purge protected by explicit confirmation string (`CONFIRM_ADMIN_GLOBAL_PURGE`).
- **Unified AI Provider Abstraction**: Polymorphic engine supporting **Google Gemini**, **NVIDIA NIM**, and **OpenAI**.
- **Strict Vector Space Isolation**: Prevents cross-provider vector contamination by enforcing provider identity and dimension checking:
  - **Google Gemini**: 768 dimensions (`gemini-embedding-2`, `text-embedding-004`)
  - **NVIDIA NIM**: 2048 dimensions (`nvidia/llama-nemotron-embed-1b-v2`)
  - **OpenAI**: 1536 dimensions (`text-embedding-3-small`, `text-embedding-ada-002`)
- **Deterministic Duplicate Protection**: Generates deterministic UUIDv5 chunk identifiers derived from owner, provider, document ID, and content metadata, enabling idempotent re-indexing.
- **Prompt Trust Boundary**: Enforces prompt isolation where all retrieved context snippets are encapsulated as untrusted passive reference data to prevent prompt injection.
- **Citation Back-References**: Responses return structured source citations containing chunk IDs, file origins, content snippets, and Cosine relevance scores.
- **Session Authentication Gate**: Application access key verification using constant-time hashing (`hmac.compare_digest`) and dedicated cryptographic `SESSION_SIGNING_KEY` issuing a signed `p06_session` HttpOnly cookie or accepting Bearer tokens.

---

## 🧠 Supported AI Providers & Vector Specifications

| Provider | Canonical ID | Chat Completion Models | Embedding Models | Vector Dimension | Qdrant Collection |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Google Gemini** | `gemini` | `gemini-2.5-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash` | `gemini-embedding-2`, `text-embedding-004` | **768** | `p06_gemini_embedding_2_768` |
| **NVIDIA NIM** | `nvidia_nim` | `nvidia/nemotron-3-super-120b-a12b`, `meta/llama-3.1-8b-instruct`, `meta/llama-3.1-70b-instruct` | `nvidia/llama-nemotron-embed-1b-v2` | **2048** | `p06_nvidia_llama_nemotron_embed_1b_v2_2048` |
| **OpenAI** | `openai` | `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo` | `text-embedding-3-small`, `text-embedding-ada-002` | **1536** | `p06_openai_text_embedding_3_small` |

---

## 📖 API Documentation & Endpoints

Interactive documentation is available at runtime:
- **Swagger UI**: [`/docs`](file:///docs) (or versioned alias `/api/v1/docs`)
- **ReDoc UI**: [`/redoc`](file:///redoc) (or versioned alias `/api/v1/redoc`)
- **OpenAPI Schema**: [`/openapi.json`](file:///openapi.json) (or versioned alias `/api/v1/openapi.json`)

### Endpoint Reference

| Method | Path | Auth Required | Role | Description |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` / `/api/v1/health` | No | Public | Service liveness and operational health probe. |
| `GET` | `/health/providers` | No | Public | Real-time connectivity and credential health for all AI providers. |
| `POST` | `/api/v1/auth/login` | No | Public | Authenticates `access_key`, sets `username`, and issues `p06_session` cookie. |
| `GET` | `/api/v1/auth/google/config`| No | Public | Returns Google OAuth client ID and redirect URI configuration. |
| `POST` | `/api/v1/auth/google/callback`| No | Public | Exchanges Google OAuth authorization code and provisions session. |
| `GET` | `/api/v1/auth/status` | No | Public | Returns session authentication status and authenticated identity. |
| `POST` | `/api/v1/auth/logout` | No | Public | Clears active `p06_session` cookie. |
| `GET` | `/api/v1/providers` | **Yes** | User / Admin | Returns sanitized metadata and supported models for all AI providers. |
| `GET` | `/api/v1/documents` | **Yes** | User / Admin | Lists documents owned by the authenticated user with Drive sync status. |
| `DELETE`| `/api/v1/documents/{doc_id}` | **Yes** | User / Admin | Deletes a specific document, vector chunks, and associated Drive file. |
| `DELETE`| `/api/v1/documents` | **Yes** | User / Admin | Clears all documents and vector chunks owned by the authenticated user. |
| `POST` | `/api/v1/documents/ingest` | **Yes** | User / Admin | Ingests PDF/TXT files, binds ownership, syncs Drive, and indexes vectors. |
| `POST` | `/api/v1/rag/query` | **Yes** | User / Admin | Decoupled semantic question answering with grounded citations. |
| `GET` | `/api/v1/rag/stats` | **Yes** | User / Admin | Returns indexed vector counts, dimensions, and store status for user. |
| `GET` | `/api/v1/admin/overview` | **Yes** | **Admin Only**| Aggregate document, chunk, user, and storage telemetry across all tenants. |
| `GET` | `/api/v1/admin/diagnostics` | **Yes** | **Admin Only**| Detailed runtime diagnostics and live provider connectivity status. |
| `POST` | `/api/v1/admin/vectors/clear`| **Yes** | **Admin Only**| Privileged global vector and metadata purge with confirmation phrase. |
| `DELETE`| `/api/v1/rag/index` | **Yes** | **Admin Only**| Administrative vector clear for specific provider. |

---

### Request & Response Examples

#### 1. Provider Health Check (`GET /health/providers`)
```bash
curl -s "http://127.0.0.1:8000/health/providers"
```
```json
{
  "status": "healthy",
  "providers": [
    {
      "provider_id": "openai",
      "name": "OpenAI",
      "configured": true,
      "status": "connected",
      "message": "Successfully connected to OpenAI API.",
      "latency_ms": 142.3
    },
    {
      "provider_id": "gemini",
      "name": "Google Gemini",
      "configured": true,
      "status": "connected",
      "message": "Successfully connected to Google Gemini API.",
      "latency_ms": 118.5
    },
    {
      "provider_id": "nvidia_nim",
      "name": "NVIDIA NIM",
      "configured": false,
      "status": "not_configured",
      "message": "NVIDIA_API_KEY is not configured.",
      "latency_ms": null
    }
  ],
  "timestamp": "2026-10-04T12:00:00Z"
}
```

#### 2. Decoupled Semantic Query (`POST /api/v1/rag/query`)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/rag/query" \
  -b cookies.txt \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What encryption standard is required for log storage?",
    "embedding_provider_id": "gemini",
    "embedding_model": "gemini-embedding-2",
    "chat_provider_id": "openai",
    "chat_model": "gpt-4o",
    "top_k": 4,
    "similarity_threshold": 0.25
  }'
```
```json
{
  "answer": "According to Section 4.2 of the Security Policy, all enterprise log archives must be encrypted at rest using AES-256.",
  "sources": [
    {
      "chunk_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "content": "Section 4.2: Data at Rest. All enterprise log archives must be encrypted using AES-256 with rotation every 90 days.",
      "score": 0.8912,
      "metadata": {
        "filename": "security_policy.pdf",
        "chunk_index": 3
      }
    }
  ],
  "provider": "OpenAI",
  "provider_id": "openai",
  "chat_model": "gpt-4o",
  "chat_provider": "OpenAI",
  "chat_provider_id": "openai",
  "embedding_provider": "Google Gemini",
  "embedding_provider_id": "gemini",
  "embedding_model": "gemini-embedding-2",
  "retrieved_count": 1
}
```

#### 3. Privileged Global Vector Purge (`POST /api/v1/admin/vectors/clear`)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/admin/vectors/clear" \
  -b cookies.txt \
  -H "Content-Type: application/json" \
  -d '{
    "confirmation": "CONFIRM_ADMIN_GLOBAL_PURGE"
  }'
```
```json
{
  "status": "success",
  "message": "Global purge complete. Purged 14 document(s) and 182 chunk(s).",
  "purged_documents": 14,
  "purged_chunks": 182,
  "provider_id": null
}
```

---

## 🔒 Security & Hardening

1. **Dedicated Session Token Signing**: `SESSION_SIGNING_KEY` is completely isolated from `APP_ACCESS_KEY` and `ADMIN_ACCESS_KEY`. Even if regular users possess the app key, they cannot forge admin tokens or modify signed claims.
2. **Encrypted Token Vault**: Google OAuth refresh and access tokens are encrypted at rest with Fernet symmetric encryption derived from server secrets.
3. **Prompt Trust Isolation**: Context chunks are strictly treated as untrusted reference data in the prompt template.
4. **Distroless & Nonroot Execution**: Runs under Debian Distroless with nonroot UID 10001.

---

## 🛠️ Local Development & Testing

```bash
# 1. Setup Virtual Environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run Test Suite (162 unit & integration tests)
pytest -v

# 3. Build Frontend
cd frontend
npm install
npm run build
cd ..

# 4. Start Server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
