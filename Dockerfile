# ──────────────────────────────────────────────────────────────────────────
# Agentic RAG System — Dockerfile
# ──────────────────────────────────────────────────────────────────────────
# Build:   docker build -t agentic-rag .
# Run:     docker run -p 8501:8501 --env-file .env agentic-rag
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────
# gcc/g++   : required by some native Python packages (e.g. rank-bm25)
# libgomp1  : OpenMP runtime for HuggingFace / PyTorch CPU
# git       : required by GitPython for repo cloning
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
# Copy requirements first to maximise Docker layer caching — only invalidated
# when requirements.txt changes, not when source code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ── Application source ─────────────────────────────────────────────────────
COPY . .

# ── Runtime directories ────────────────────────────────────────────────────
# Created here so they exist when the container starts, even if no volumes
# are mounted. Actual data will flow in via docker-compose volume mounts.
RUN mkdir -p local_qdrant uploaded_docs cloned_repos

# ── Non-root user (security hardening) ────────────────────────────────────
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Port ───────────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ───────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
    || exit 1

# ── Entry point ────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=true"]
