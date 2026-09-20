# ==============================================================================
# Multi-Stage Production Dockerfile for Unified Enterprise RAG System (P06)
# Target Architecture: Linux amd64 / arm64 (OCI Ampere A1 Compatible)
# Runtime: Python 3.12 on Debian 12 (Bookworm) Slim
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

# Install build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install to isolated prefix
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


# ------------------------------------------------------------------------------
# Stage 2: Hardened Minimal Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/install/lib/python3.12/site-packages:/app" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Create non-root system user and group (appuser: 10001)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -d /app appuser

# Copy installed Python packages from builder stage
COPY --from=builder /install /install

# Copy application source code with non-root ownership
COPY --chown=appuser:appgroup app.py .
COPY --chown=appuser:appgroup config/ ./config/
COPY --chown=appuser:appgroup providers/ ./providers/
COPY --chown=appuser:appgroup rag/ ./rag/
COPY --chown=appuser:appgroup utils/ ./utils/

# Switch to non-root execution
USER appuser:appgroup

# Expose Streamlit application port
EXPOSE 8501

# Native HTTP healthcheck (no curl required)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]

# Production Entrypoint
CMD ["streamlit", "run", "app.py"]
