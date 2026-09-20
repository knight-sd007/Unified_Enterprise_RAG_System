# ==============================================================================
# Multi-Stage Hardened Production Dockerfile for Unified Enterprise RAG System (P06)
# Target Architecture: Linux amd64 / arm64 (OCI Ampere A1 Compatible)
# Builder: Python 3.12 on Debian 12 (Bookworm) Slim
# Runtime: Hardened Minimal Distroless C/C++ Debian 12 (Non-Root UID: 65532)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build Dependencies & Prepare Wheel Packages
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

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
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ------------------------------------------------------------------------------
# Stage 2: Hardened Minimal Production Runtime (Distroless C/C++ Debian 12)
# ------------------------------------------------------------------------------
FROM gcr.io/distroless/cc-debian12:nonroot AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/usr/local/lib/python3.12/site-packages:/app" \
    LD_LIBRARY_PATH="/usr/local/lib:/usr/lib" \
    PATH="/usr/local/bin:$PATH" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Copy CPython 3.12 runtime binaries, dynamic libraries, and standard library from builder
COPY --from=builder /usr/local/bin/python3* /usr/local/bin/
COPY --from=builder /usr/local/lib/libpython3* /usr/local/lib/
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12

# Copy required dynamic shared libraries for SQLite, ctypes, bz2, lzma, and zlib
COPY --from=builder /usr/lib/*-linux-gnu*/libsqlite3.so.0* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu*/libffi.so.8* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu*/libbz2.so.1.0* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu*/liblzma.so.5* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu*/libz.so.1* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu*/libssl.so.3* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu*/libcrypto.so.3* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu*/libexpat.so.1* /usr/lib/

# Copy isolated production Python packages and Streamlit executable from builder
COPY --from=builder /install/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /install/bin/streamlit /usr/local/bin/streamlit

# Copy application source code with nonroot ownership (UID: 65532)
COPY --chown=65532:65532 app.py .
COPY --chown=65532:65532 config/ ./config/
COPY --chown=65532:65532 providers/ ./providers/
COPY --chown=65532:65532 rag/ ./rag/
COPY --chown=65532:65532 utils/ ./utils/

# Enforce non-root execution (built-in distroless nonroot user)
USER nonroot:nonroot

# Expose Streamlit application port
EXPOSE 8501

# Exec-form HTTP healthcheck without shell dependency
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]

# Production Entrypoint using verified CPython 3.12 interpreter
CMD ["/usr/local/bin/python3", "-m", "streamlit", "run", "app.py"]
