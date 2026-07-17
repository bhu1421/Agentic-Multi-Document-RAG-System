import time
import os
import functools
from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from qdrant_client.http import models
from backend import config
from backend.logger import get_logger

logger = get_logger(__name__)


def _get_device() -> str:
    """Auto-detect the best available hardware accelerator."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def should_use_reranker() -> bool:
    """Determine whether to run the cross-encoder reranker.

    Priority order:
    1. ENABLE_RERANKER env var explicitly set to true/1/yes  → always rerank
    2. ENABLE_RERANKER env var explicitly set to false/0/no  → never rerank
    3. No env var set → rerank only when a GPU/MPS is available (CPU is too slow)

    The sidebar toggle sets os.environ["ENABLE_RERANKER"] at runtime, so
    users can toggle reranking on/off without restarting the app.
    """
    val = os.getenv("ENABLE_RERANKER", "").lower()
    if val in {"1", "true", "yes"}:
        return True
    if val in {"0", "false", "no"}:
        return False
    # Default: only rerank when hardware acceleration is available
    return _get_device() != "cpu"


@functools.lru_cache(maxsize=1)
def get_reranker():
    """Load the cross-encoder reranker model once and reuse it.

    Design decision: two-stage retrieval
    Stage 1 (bi-encoder / MiniLM) — fast approximate nearest-neighbour search.
             Retrieves config.TOP_K candidates in milliseconds.
    Stage 2 (cross-encoder / BGE) — precise relevance scoring.
             Scores (query, document) pairs together; much more accurate but
             O(n) with document count — only applied to the top candidates.
    """
    device = _get_device()
    logger.info("[Device] %s running on: %s", config.RERANKER_MODEL, device.upper())
    t = time.time()
    model = HuggingFaceCrossEncoder(
        model_name=config.RERANKER_MODEL,
        model_kwargs={"device": device},
    )
    logger.info("[Cache] Reranker loaded in %.1fs", time.time() - t)
    return model


def _build_qdrant_filter(target_sources=None, metadata_filters=None):
    """Build a Qdrant Filter combining source targets and metadata conditions.

    Source targets use a `should` (OR) filter — match ANY of the listed sources.
    Metadata conditions use `must` (AND) — ALL conditions must match.
    Both are wrapped in a top-level `must` so they combine with AND semantics.
    """
    must_conditions = []

    if target_sources:
        source_filter = models.Filter(
            should=[
                models.FieldCondition(key="metadata.source", match=models.MatchValue(value=src))
                for src in target_sources
            ]
        )
        must_conditions.append(source_filter)

    if metadata_filters:
        for key, value in metadata_filters.items():
            if key == "source" or value is None:
                continue
            must_conditions.append(
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value),
                )
            )

    if must_conditions:
        return models.Filter(must=must_conditions)
    return None


def build_retrieval_pipeline(store, target_sources=None, metadata_filters=None):
    """Build the Qdrant MMR retrieval pipeline.

    MMR (Maximal Marginal Relevance) balances relevance with diversity:
    it penalises chunks that are too similar to already-selected chunks,
    reducing redundancy in the context window.

    config.TOP_K  — how many final candidates to return
    config.FETCH_K — how many candidates MMR considers before pruning to TOP_K
    """
    search_kwargs = {"k": config.TOP_K, "fetch_k": config.FETCH_K}

    qdrant_filter = _build_qdrant_filter(target_sources, metadata_filters)
    if qdrant_filter:
        search_kwargs["filter"] = qdrant_filter
        logger.info(
            "[Retriever] Qdrant filter: sources=%s metadata=%s",
            target_sources, metadata_filters,
        )

    return store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)


def expand_to_parent_context(matched_docs, store, top_n_parents=config.MAX_PARENTS, user_id: str | None = None):
    """Hierarchical retrieval: given exact chunk matches, fetch all sibling chunks.

    Problem solved: a matched chunk might say "He signed the contract on page 4."
    Without context, "He" is ambiguous. By fetching all chunks sharing the same
    parent_id, we reconstruct the full document section and give the LLM
    complete, coherent context.

    Implementation: every chunk stores a parent_id set at indexing time
    (see chunker.py). We use Qdrant's scroll API to fetch all chunks with
    matching parent_id, then sort them by page/chunk_id to restore reading order.
    """
    if not store or not matched_docs:
        return matched_docs

    parent_ids = []
    seen_parents = set()
    for d in matched_docs:
        pid = d.metadata.get("parent_id")
        if pid and pid not in seen_parents:
            seen_parents.add(pid)
            parent_ids.append(pid)
            if len(parent_ids) >= top_n_parents:
                break

    if not parent_ids:
        return matched_docs

    logger.info("[HierarchicalRetrieval] Expanding context for %d parent documents", len(parent_ids))

    expanded_docs = []
    seen_chunks = set()
    client = store.client
    collection_name = store.collection_name

    for pid in parent_ids:
        try:
            records, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.parent_id",
                            match=models.MatchValue(value=pid),
                        )
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False,
            )

            siblings = [
                Document(
                    page_content=r.payload.get("page_content", ""),
                    metadata=r.payload.get("metadata", {}),
                )
                for r in records
            ]

            # Restore reading order within the parent document
            siblings.sort(key=lambda d: (d.metadata.get("page", 0), d.metadata.get("chunk_id", "")))

            for doc in siblings:
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id is None or chunk_id not in seen_chunks:
                    if chunk_id is not None:
                        seen_chunks.add(chunk_id)
                    expanded_docs.append(doc)

        except Exception as exc:
            logger.warning("[HierarchicalRetrieval] Failed to expand parent_id %s: %s", pid, exc)

    logger.info(
        "[HierarchicalRetrieval] Expanded %d initial chunks → %d full-document chunks",
        len(matched_docs), len(expanded_docs),
    )

    # Cap to prevent blowing up the reranker / context window
    return expanded_docs[:config.MAX_EXPANDED_CHUNKS]


# ──────────────────────────────────────────────
# Hybrid Retrieval (Dense + BM25)
# ──────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    store,
    target_sources: list[str] | None = None,
    user_id: str | None = None,
) -> list[Document]:
    """Run dense + BM25 retrieval and fuse results with Reciprocal Rank Fusion.

    Pipeline:
    1. Dense search (Qdrant MMR) → TOP_K candidates
    2. BM25 search (keyword matching) → BM25_TOP_K candidates
    3. Reciprocal Rank Fusion → merged ranked list

    If BM25 fails for any reason, falls back to dense-only results.

    Args:
        query: The user's search query.
        store: The Qdrant vector store instance.
        target_sources: Optional list of source names to filter by.
        user_id: Optional user ID to scope the search.

    Returns:
        A fused, ranked list of Document objects.
    """
    from backend.bm25 import bm25_search, reciprocal_rank_fusion

    metadata_filters = {"user_id": user_id} if user_id else None

    # ── Dense retrieval (existing MMR pipeline) ───────────────────────────────
    pipeline = build_retrieval_pipeline(store, target_sources, metadata_filters)
    dense_docs = pipeline.invoke(query)
    logger.info("[HybridSearch] Dense retrieved %d docs", len(dense_docs))

    # ── BM25 retrieval ────────────────────────────────────────────────────────
    try:
        bm25_docs = bm25_search(
            query=query,
            store=store,
            target_sources=target_sources,
            user_id=user_id,
        )
        logger.info("[HybridSearch] BM25 retrieved %d docs", len(bm25_docs))
    except Exception as exc:
        logger.warning("[HybridSearch] BM25 search failed (%s) — using dense only", exc)
        return dense_docs

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────
    if not bm25_docs:
        return dense_docs
    if not dense_docs:
        return bm25_docs

    fused = reciprocal_rank_fusion(dense_docs, bm25_docs)
    return fused

