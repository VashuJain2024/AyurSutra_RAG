# ── Base image ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

# HF Spaces runs as a non-root user; set this to avoid permission issues
RUN useradd -m -u 1000 appuser

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps (no streamlit needed) ────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir \
    langchain>=0.3.0 \
    langchain-community>=0.3.0 \
    langchain-huggingface \
    langchain-groq \
    faiss-cpu \
    sentence-transformers \
    pypdf \
    python-dotenv \
    fastapi \
    "uvicorn[standard]"

# ── Copy app source ───────────────────────────────────────────────────────────
COPY api.py .
COPY rag_pipeline.py .

# ── Copy the pre-built vector DB (already in your repo) ──────────────────────
COPY vector_db/ ./vector_db/

# ── Switch to non-root user ───────────────────────────────────────────────────
RUN chown -R appuser:appuser /app
USER appuser

# ── HuggingFace Spaces uses port 7860 ────────────────────────────────────────
ENV PORT=7860
EXPOSE 7860

# ── Start FastAPI via Uvicorn ─────────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]