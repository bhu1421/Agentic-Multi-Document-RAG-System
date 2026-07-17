"""
BM25 Sparse Retrieval & Reciprocal Rank Fusion for Hybrid Search.

This module provides two capabilities:
    bm25_search()             — keyword-based retrieval using BM25 (Okapi BM25)
    reciprocal_rank_fusion()  — merges dense + sparse ranked lists via RRF

Why BM25?
    Dense (semantic) search encodes meaning but struggles with exact keywords
    like API names, version numbers, function names, and IDs.  BM25 excels at
    these because it matches on exact token overlap.  Combining both gives the
    best of both worlds — semantic understanding AND keyword precision.

Design decisions:
    - BM25 index is built on-the-fly from Qdrant's stored documents.
      This avoids a separate persistence layer and stays consistent with
      whatever is currently indexed.  Performant up to ~50k chunks.
    - Tokenisation is simple (lowercase + split on non-alphanumeric).  This is
      intentionally basic — BM25's strength is exact matching, not NLP.
    - RRF uses rank-based fusion (no score normalisation needed between dense
      and sparse), which is the industry standard (Elasticsearch, Azure AI Search).
"""

import re
import time
from langchain_core.documents import Document
from qdrant_client.http import models
from backend import config
from backend.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Tokeniser
# ──────────────────────────────────────────────

_RE_TOKEN = re.compile(r"[a-zA-Z0-9]+")


def _tokenise(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric boundaries.

    Intentionally simple: BM25's value is exact token matching, so we keep
    tokens like 'gpt', '4', '1', 'api', 'v2' as separate searchable units.
    """
    return _RE_TOKEN.findall(text.lower())


# ──────────────────────────────────────────────
# BM25 Search
# ──────────────────────────────────────────────

def bm25_search(
    query: str,
    store,
    target_sources: list[str] | None = None,
    user_id: str | None = None,
    top_k: int | None = None,
) -> list[Document]:
    """Run BM25 keyword search over all documents in the Qdrant collection.

    Steps:
    1. Scroll ALL documents from Qdrant (with optional source/user filters)
    2. Tokenise each document's page_content
    3. Build an in-memory BM25 index
    4. Score the query against the corpus
    5. Return the top-k documents, ranked by BM25 score

    Args:
        query: The user's search query.
        store: The Qdrant vector store instance.
        target_sources: Optional list of source names to filter by.
        user_id: Optional user ID to scope the search.
        top_k: Number of results to return (defaults to config.BM25_TOP_K).

    Returns:
        A ranked list of Document objects.
    """
    from rank_bm25 import BM25Okapi

    if top_k is None:
        top_k = config.BM25_TOP_K

    t = time.time()
    client = store.client
    collection_name = store.collection_name

    # ── Build scroll filter ───────────────────────────────────────────────────
    must_conditions = []
    if target_sources:
        must_conditions.append(
            models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=src),
                    )
                    for src in target_sources
                ]
            )
        )
    if user_id:
        must_conditions.append(
            models.FieldCondition(
                key="metadata.user_id",
                match=models.MatchValue(value=user_id),
            )
        )

    scroll_filter = models.Filter(must=must_conditions) if must_conditions else None

    # ── Scroll all documents from Qdrant ──────────────────────────────────────
    all_records = []
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_records.extend(records)
        if next_offset is None:
            break
        offset = next_offset

    if not all_records:
        logger.info("[BM25] No documents in collection to search")
        return []

    # ── Build Document objects + tokenised corpus ─────────────────────────────
    documents: list[Document] = []
    corpus: list[list[str]] = []

    for record in all_records:
        content = record.payload.get("page_content", "")
        metadata = record.payload.get("metadata", {})
        documents.append(Document(page_content=content, metadata=metadata))
        corpus.append(_tokenise(content))

    # ── Build BM25 index and score ────────────────────────────────────────────
    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenise(query)
    scores = bm25.get_scores(query_tokens)

    # ── Rank and return top-k ─────────────────────────────────────────────────
    scored_docs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    top_docs = [doc for doc, score in scored_docs[:top_k] if score > 0]

    logger.info(
        "[BM25] Searched %d documents → %d results in %.2fs",
        len(all_records), len(top_docs), time.time() - t,
    )
    return top_docs


# ──────────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_docs: list[Document],
    bm25_docs: list[Document],
    k: int | None = None,
) -> list[Document]:
    """Merge two ranked document lists using Reciprocal Rank Fusion (RRF).

    RRF formula:  score(d) = Σ  1 / (k + rank_i(d))
    where rank_i(d) is the 1-based rank of document d in list i.

    Why RRF?
    - Rank-based: no need to normalise scores across different retrieval systems
    - Proven: used by Elasticsearch, Azure AI Search, and major production systems
    - Robust: a document ranked highly in EITHER list will surface in the fusion

    Deduplication is by chunk_id metadata. If a document appears in both lists,
    its RRF scores from both lists are summed (rewarding agreement).

    Args:
        dense_docs: Ranked results from dense (semantic) retrieval.
        bm25_docs: Ranked results from BM25 (keyword) retrieval.
        k: The RRF constant (defaults to config.RRF_K). Higher k reduces the
           influence of high ranks; 60 is the standard value.

    Returns:
        A single merged and re-ranked list of Documents.
    """
    if k is None:
        k = config.RRF_K

    # Map chunk_id → (document, cumulative RRF score)
    doc_map: dict[str, Document] = {}
    score_map: dict[str, float] = {}

    def _add_scores(docs: list[Document]):
        for rank, doc in enumerate(docs, start=1):
            chunk_id = doc.metadata.get("chunk_id", f"unknown_{id(doc)}")
            rrf_score = 1.0 / (k + rank)

            if chunk_id in score_map:
                score_map[chunk_id] += rrf_score
            else:
                score_map[chunk_id] = rrf_score
                doc_map[chunk_id] = doc

    _add_scores(dense_docs)
    _add_scores(bm25_docs)

    # Sort by cumulative RRF score (descending)
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    fused_docs = [doc_map[chunk_id] for chunk_id, _ in ranked]

    logger.info(
        "[RRF] Fused %d dense + %d BM25 → %d unique documents",
        len(dense_docs), len(bm25_docs), len(fused_docs),
    )
    return fused_docs
