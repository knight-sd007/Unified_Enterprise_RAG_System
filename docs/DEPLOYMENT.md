# Production Deployment Runbook — Unified Enterprise RAG System (P06)

---

## 1. Overview & Architecture

The Unified Enterprise RAG System (Portfolio 06) is containerized and deployed using an automated CI/CD pipeline integrated with Docker Hub, target host execution, and Cloudflare Tunnel for secure HTTPS ingress:

```text
GitHub (Push / Tag)
       │
       ▼
Jenkins CI/CD Pipeline
  ├── 1. Gitleaks Secret Detection
  ├── 2. Pytest Automated Test Execution (Python 3.12)
  ├── 3. Multi-Arch Docker Build (linux/arm64, linux/amd64)
  ├── 4. Trivy Container Vulnerability Scan
  ├── 5. Docker Hub Publish (Immutable Git-SHA Tag + Latest)
  ├── 6. Target Deployment (docker compose up -d)
  └── 7. 3-Layer Post-Deployment Verification
              │
              ▼
   OCI Ampere A1 Host (:8006)
              │
       Cloudflared Tunnel
              │
              ▼
    Public Ingress (https://rag.vaikuntrix.in)
```

---

## 2. Container Configuration

### A. Docker Image Specification
- **Base Image:** `python:3.12-slim-bookworm`
- **Build Strategy:** Multi-stage build (`builder` $\to$ `runner`)
- **Execution User:** Non-root system user `appuser` (UID: 10001, GID: 10001)
- **Exposed Port:** `8501` (Internal Streamlit port)
- **Healthcheck Endpoint:** `http://127.0.0.1:8501/_stcore/health`
- **Zero Embedded Secrets:** `.dockerignore` strictly excludes `.git`, `.venv`, `.env*`, and credentials.

### B. Local Docker Commands

**1. Build local container image:**
```bash
docker build -t knightprime007/unified-enterprise-rag-system:local .
```

**2. Run container with local environment variables:**
```bash
docker run -d \
  --name p06_rag_system \
  -p 8006:8501 \
  --env-file .env \
  knightprime007/unified-enterprise-rag-system:local
```

**3. Verify container health status:**
```bash
docker ps --filter name=p06_rag_system
curl -I http://127.0.0.1:8006/_stcore/health
```

---

## 3. Runtime Environment Configuration

All credentials and runtime settings must be injected from the host's `/opt/projects/enterprise-rag/.env` file. Never bake credentials into container images.

| Variable | Required | Description | Example / Default |
| :--- | :--- | :--- | :--- |
| `APP_ACCESS_KEY` | Yes | Application login access key | (Set strong production key) |
| `AI_PROVIDER` | Yes | Active provider (`gemini`, `nvidia_nim`, `openai`) | `gemini` |
| `GEMINI_API_KEY` | Conditional | Google Gemini API credentials | `AIzaSy...` |
| `GEMINI_CHAT_MODEL` | No | Gemini chat model | `gemini-2.5-flash` |
| `GEMINI_EMBEDDING_MODEL`| No | Gemini embedding model | `text-embedding-004` |
| `NVIDIA_API_KEY` | Conditional | NVIDIA NIM API credentials | `nvapi-...` |
| `NVIDIA_BASE_URL` | No | NVIDIA NIM base endpoint | `https://integrate.api.nvidia.com/v1` |
| `OPENAI_API_KEY` | Conditional | OpenAI API credentials | `sk-...` |
| `QDRANT_URL` | Conditional | Qdrant Cloud cluster endpoint | `https://<cluster>.qdrant.tech:6333` |
| `QDRANT_API_KEY` | Conditional | Qdrant Cloud API token | (Secret token) |

---

## 4. Jenkins CI/CD Setup

### A. Jenkins Credentials Requirements
The Jenkins pipeline references credentials configured in Jenkins Credential Manager:

- **`docker-hub-credentials`**: Username and Password credentials with Docker Hub publish access.

### B. Pipeline Execution Workflow
1. **Gitleaks Secret Scan**: Scans workspace for exposed API keys or tokens.
2. **Pytest Verification**: Runs the complete unit and integration test suite inside a `python:3.12-slim-bookworm` container.
3. **Multi-Arch Build**: Compiles Docker image tagged with immutable `${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${GIT_SHA}`.
4. **Trivy Vulnerability Scan**: Assesses container image vulnerabilities and fails build if `HIGH` or `CRITICAL` CVEs exist.
5. **Docker Hub Publish**: Pushes immutable git-sha tag and `:latest`.
6. **Deployment Trigger**: Pulls and deploys updated image on target host using `docker compose`.
7. **3-Layer Post-Deployment Verification**:
   - **Layer 1 (Internal Readiness):** Probes `http://127.0.0.1:8006/_stcore/health` and verifies container image SHA.
   - **Layer 2 (Tunnel Health):** Confirms `cloudflared` process is active.
   - **Layer 3 (Public Route):** Probes `https://rag.vaikuntrix.in` to confirm edge ingress.

---

## 5. Cloudflare Tunnel Ingress

Ingress configuration template is maintained at `deploy/cloudflared/config.yml`.

### Ingress Mapping:
```yaml
ingress:
  - hostname: rag.vaikuntrix.in
    service: http://127.0.0.1:8006
  - service: http_status:404
```

---

## 6. Rollback Procedure

If a production regression occurs:

1. Identify the last known good immutable image tag from Docker Hub (e.g. `knightprime007/unified-enterprise-rag-system:<previous-git-sha>`).
2. Run rollback on target host:
   ```bash
   cd /opt/projects/enterprise-rag
   P06_IMAGE="knightprime007/unified-enterprise-rag-system:<previous-git-sha>" \
     docker compose --env-file .env -f docker-compose.yml up -d
   ```
3. Verify recovery via `http://127.0.0.1:8006/_stcore/health` and public domain.
