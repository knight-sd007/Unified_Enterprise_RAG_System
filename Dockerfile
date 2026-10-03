# ==============================================================================
# Multi-Stage Hardened Production Dockerfile for Unified Enterprise RAG System (P06)
# Target Architecture: Linux amd64 / arm64 (OCI Ampere A1 Compatible)
# Stage 1: Python Dependencies Builder (Debian 13 Trixie Slim)
# Stage 2: React Frontend Builder (Node.js 20 Slim)
# Stage 3: Minimal Hardened Production Runner (Distroless Nonroot UID: 65532)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build Python Dependencies
# ------------------------------------------------------------------------------
FROM python:3.12-slim-trixie AS python-builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install required build tools and C-extension shared headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libsqlite3-0 \
    libffi-dev \
    libbz2-1.0 \
    liblzma5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ------------------------------------------------------------------------------
# Stage 2: Build React + Vite Frontend Assets
# ------------------------------------------------------------------------------
FROM node:20-slim AS frontend-builder

WORKDIR /frontend

COPY frontend/package.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build


# ------------------------------------------------------------------------------
# Stage 3: Hardened Minimal Production Runtime (Distroless C/C++ Debian 13)
# ------------------------------------------------------------------------------
FROM gcr.io/distroless/cc-debian13:nonroot AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/usr/local/lib/python3.12/site-packages:/app" \
    LD_LIBRARY_PATH="/usr/local/lib:/usr/lib" \
    PATH="/usr/local/bin:$PATH"

# Copy CPython 3.12 runtime binaries, dynamic libraries, and standard library from builder
COPY --from=python-builder /usr/local/bin/python3* /usr/local/bin/
COPY --from=python-builder /usr/local/lib/libpython3* /usr/local/lib/
COPY --from=python-builder /usr/local/lib/python3.12 /usr/local/lib/python3.12

# Copy required dynamic shared libraries for SQLite, ctypes, bz2, lzma, and zlib
COPY --from=python-builder /usr/lib/*-linux-gnu*/libsqlite3.so.0* /usr/lib/
COPY --from=python-builder /usr/lib/*-linux-gnu*/libffi.so.8* /usr/lib/
COPY --from=python-builder /usr/lib/*-linux-gnu*/libbz2.so.1.0* /usr/lib/
COPY --from=python-builder /usr/lib/*-linux-gnu*/liblzma.so.5* /usr/lib/
COPY --from=python-builder /usr/lib/*-linux-gnu*/libz.so.1* /usr/lib/
COPY --from=python-builder /usr/lib/*-linux-gnu*/libssl.so.3* /usr/lib/
COPY --from=python-builder /usr/lib/*-linux-gnu*/libcrypto.so.3* /usr/lib/
COPY --from=python-builder /usr/lib/*-linux-gnu*/libexpat.so.1* /usr/lib/

# Copy isolated production Python packages from builder
COPY --from=python-builder /install/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=python-builder /install/bin/uvicorn /usr/local/bin/uvicorn

# Copy backend application source code with nonroot ownership (UID: 65532)
COPY --chown=65532:65532 api/ ./api/
COPY --chown=65532:65532 config/ ./config/
COPY --chown=65532:65532 providers/ ./providers/
COPY --chown=65532:65532 rag/ ./rag/
COPY --chown=65532:65532 utils/ ./utils/

# Copy compiled React frontend distribution assets
COPY --from=frontend-builder --chown=65532:65532 /frontend/dist ./frontend/dist

# Enforce non-root execution (built-in distroless nonroot user)
USER nonroot:nonroot

# Expose FastAPI application port
EXPOSE 8000

# Exec-form HTTP healthcheck without shell dependency
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)"]

# Production Entrypoint using verified CPython 3.12 interpreter
CMD ["/usr/local/bin/python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
