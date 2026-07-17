"""
Central configuration for the Agentic RAG system.

All tunable constants live here. To change the LLM, embedding model,
chunk size, or retrieval parameters — edit this file only.
No other file needs to change.

Design decision: A flat config module (vs. pydantic-settings or YAML)
was chosen because it requires zero extra dependencies and is trivially
importable from anywhere in the package.
"""

import os

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL: str = "llama-3.1-8b-instant"   # Groq-hosted Llama model
LLM_TEMPERATURE: float = 0.1               # Near-deterministic for routing + RAG
LLM_REQUEST_TIMEOUT: int = 30             # Seconds before a Groq request is aborted
LLM_MAX_RETRIES: int = 2                  # Retry transient network errors

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Runs locally; no API key needed
EMBEDDING_DIM: int = 384                    # Must match the model above exactly

# ── Reranker ──────────────────────────────────────────────────────────────────
RERANKER_MODEL: str = "BAAI/bge-reranker-large"  # Cross-encoder; GPU-recommended

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 800       # Characters per chunk — balances context vs. precision
CHUNK_OVERLAP: int = 150    # Prevents answers from being split across chunk boundaries

# ── Retrieval ─────────────────────────────────────────────────────────────────
# Two-stage retrieval: broad MMR pass first, then cross-encoder reranking.
TOP_K: int = 30             # Candidates retrieved from Qdrant (wide net for recall)
FETCH_K: int = 60           # Candidates considered by MMR before selecting TOP_K
MAX_RERANKED: int = 8       # Final docs kept after reranking (fed to the LLM)
MAX_PARENTS: int = 5        # Max parent documents expanded in hierarchical retrieval
MAX_EXPANDED_CHUNKS: int = 50  # Hard cap on total chunks after parent expansion

# ── Answer Generation ─────────────────────────────────────────────────────────
CONTEXT_MAX_CHARS: int = 12_000   # Truncate combined context beyond this length
CHAT_HISTORY_WINDOW: int = 10     # Number of past messages passed to the LLM

# ── Vector Database ───────────────────────────────────────────────────────────
QDRANT_PATH: str = "local_qdrant"       # Directory for the on-disk Qdrant DB
COLLECTION_NAME: str = "rag_collection" # Single shared collection name

# ── File Storage ──────────────────────────────────────────────────────────────
UPLOAD_DIR: str = "uploaded_docs"       # Staging area for uploaded files
CLONED_REPOS_DIR: str = "cloned_repos"  # Staging area for cloned GitHub repos
MAX_FILE_SIZE_MB: int = 50             # Per-file upload size limit

# ── Guardrail ─────────────────────────────────────────────────────────────────
GUARDRAIL_MAX_QUERY_LENGTH: int = 5000   # Reject queries longer than this (chars)
GUARDRAIL_ENABLE_LLM_CHECK: bool = True  # Use LLM for off-topic detection

# ── Query Rewriting ───────────────────────────────────────────────────────────
RETRIEVAL_CONFIDENCE_THRESHOLD: float = 0.35  # Below this → rewrite + retry
MAX_RETRIEVAL_ATTEMPTS: int = 1               # Max rewrite retries (1 = rewrite once)

# ── Redis Cache ───────────────────────────────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL_SECONDS: int = 6 * 3600  # 6 hours
CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() in {"1", "true", "yes"}

# ── Hybrid Search (BM25 + Dense) ─────────────────────────────────────────────
ENABLE_HYBRID_SEARCH: bool = True   # Fuse BM25 keyword search with dense retrieval
BM25_TOP_K: int = 30                # Candidates returned by BM25 (matches dense TOP_K)
RRF_K: int = 60                     # Reciprocal Rank Fusion constant (standard value)
