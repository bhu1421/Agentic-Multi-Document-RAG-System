import time
import streamlit as st
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from qdrant_client.http import models

# Module-level caches
_reranker_model = None
_bm25_docs_cache = None        # cached list of Document objects
_bm25_docs_hash = None         # track when docs change (count-based)

@st.cache_resource(show_spinner=False)
def get_reranker():
    """Load the cross-encoder reranker model once and reuse it."""
    print("[Cache] Loading BAAI/bge-reranker-large (one-time)...")
    t = time.time()
    model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-large",
        model_kwargs={'device': 'cpu'}  # Force CPU to prevent meta tensor bugs
    )
    print(f"[Cache] Reranker loaded in {time.time() - t:.1f}s")
    return model

def get_cached_docs(store):
    """Cache the BM25 document list. Only re-fetch if collection size changes."""
    global _bm25_docs_cache, _bm25_docs_hash
    
    try:
        # Quick count check — avoids full scroll if nothing changed
        collection_info = store.client.get_collection("rag_collection")
        current_count = collection_info.points_count
    except Exception:
        current_count = -1
    
    if _bm25_docs_cache is not None and _bm25_docs_hash == current_count:
        print(f"[Cache] BM25 docs cache hit ({current_count} docs)")
        return _bm25_docs_cache
    
    # Cache miss — rebuild
    print(f"[Cache] Rebuilding BM25 docs cache ({current_count} docs)...")
    try:
        records, _ = store.client.scroll(
            collection_name="rag_collection",
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        _bm25_docs_cache = [
            Document(
                page_content=r.payload.get("page_content", ""),
                metadata=r.payload.get("metadata", {})
            )
            for r in records
        ]
        _bm25_docs_hash = current_count
    except Exception:
        _bm25_docs_cache = []
        _bm25_docs_hash = None
    
    return _bm25_docs_cache


def _build_qdrant_filter(target_sources=None, metadata_filters=None):
    """Build a Qdrant Filter combining source targets and metadata conditions.

    Source targets use a `should` (OR) filter — match ANY of the listed sources.
    Metadata conditions use `must` (AND) — ALL conditions must match.
    Both are wrapped in a top-level `must` so they combine with AND semantics.
    """
    must_conditions = []

    # Source filter (OR: match any of these sources)
    if target_sources:
        source_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=src)
                )
                for src in target_sources
            ]
        )
        must_conditions.append(source_filter)

    # Metadata filters (AND: all must match)
    if metadata_filters:
        for key, value in metadata_filters.items():
            if key == "source" or value is None:
                continue  # source is handled above
            must_conditions.append(
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value)
                )
            )

    if must_conditions:
        return models.Filter(must=must_conditions)
    return None


class CustomHybridRetriever:
    """Custom Reciprocal Rank Fusion (Ensemble) retriever.

    Combines dense (Qdrant MMR) and sparse (BM25) results via RRF.
    Reranking is now handled by a separate graph node, so this class
    only performs retrieval and fusion.
    """
    def __init__(self, qdrant_retriever, bm25_retriever, top_k=15):
        self.qdrant_retriever = qdrant_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k

    def _rrf(self, doc_lists, k=60):
        """Reciprocal Rank Fusion."""
        rrf_score = {}
        for doc_list in doc_lists:
            for rank, doc in enumerate(doc_list):
                # Use page_content as unique identifier for RRF
                content = doc.page_content
                if content not in rrf_score:
                    rrf_score[content] = {"doc": doc, "score": 0.0}
                rrf_score[content]["score"] += 1.0 / (rank + k)
        
        ranked_docs = sorted(rrf_score.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked_docs]

    def invoke(self, query: str):
        # 1. Fetch from Dense (Qdrant)
        qdrant_docs = self.qdrant_retriever.invoke(query)
        
        # 2. Fetch from Sparse (BM25) if available
        bm25_docs = []
        if self.bm25_retriever:
            bm25_docs = self.bm25_retriever.invoke(query)
            
        # 3. Combine via RRF
        if bm25_docs:
            fused_docs = self._rrf([qdrant_docs, bm25_docs])
        else:
            fused_docs = qdrant_docs

        # Return top_k — reranking is handled downstream by the Reranker node
        return fused_docs[:self.top_k]


def build_retrieval_pipeline(store, llm, all_docs, target_sources=None, metadata_filters=None):
    """Build the full retrieval pipeline (BM25 + Qdrant → RRF).

    Reranking has been extracted to a standalone graph node so it can
    operate on the full fused evidence (local + web + metadata-filtered).

    Args:
        store: Qdrant vector store instance.
        llm: Language model (unused here, kept for API compatibility).
        all_docs: Full list of cached documents for BM25.
        target_sources: List of source filenames to restrict search to.
        metadata_filters: Dict of metadata field conditions (file_type, page, section).
    """
    # --- Dense Retriever (Qdrant MMR) ---
    search_kwargs = {"k": 30, "fetch_k": 60}

    # Build combined Qdrant filter from sources + metadata
    qdrant_filter = _build_qdrant_filter(target_sources, metadata_filters)
    if qdrant_filter:
        search_kwargs["filter"] = qdrant_filter
        print(f"[Retriever] Qdrant filter applied: sources={target_sources}, metadata={metadata_filters}")

    qdrant_retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    # --- Sparse Retriever (BM25) ---
    bm25_retriever = None
    try:
        # Filter BM25 docs by source
        if target_sources:
            filtered_docs = [d for d in all_docs if d.metadata.get("source") in target_sources]
        else:
            filtered_docs = all_docs

        # Also apply metadata filters to BM25 docs (string comparison for type safety)
        if metadata_filters:
            for key, value in metadata_filters.items():
                if key != "source" and value is not None:
                    filtered_docs = [
                        d for d in filtered_docs
                        if str(d.metadata.get(key, "")) == str(value)
                    ]

        if filtered_docs:
            bm25_retriever = BM25Retriever.from_documents(filtered_docs)
            bm25_retriever.k = 30
    except Exception as e:
        print(f"[BM25] Initialization failed, falling back to Qdrant only: {e}")

    # Return pipeline — reranking happens in the standalone Reranker node
    return CustomHybridRetriever(
        qdrant_retriever=qdrant_retriever,
        bm25_retriever=bm25_retriever,
        top_k=15  # Return more candidates; Reranker will narrow to top 8
    )
