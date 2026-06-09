import time
import streamlit as st
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import EnsembleRetriever
from qdrant_client.http import models

# Module-level caches
_reranker_model = None
_reranker_compressor = None
_bm25_docs_cache = None        # cached list of Document objects
_bm25_docs_hash = None         # track when docs change (count-based)

@st.cache_resource(show_spinner=False)
def get_reranker():
    """Load the cross-encoder reranker model once and reuse it."""
    print("[Cache] Loading BAAI/bge-reranker-large (one-time)...")
    t = time.time()
    _reranker_model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-large",
        model_kwargs={'device': 'cpu'}  # Force CPU to prevent meta tensor bugs
    )
    _reranker_compressor = CrossEncoderReranker(model=_reranker_model, top_n=8)
    print(f"[Cache] Reranker loaded in {time.time() - t:.1f}s")
    return _reranker_compressor

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

def build_retrieval_pipeline(store, llm, all_docs, target_sources=None):
    """Build the full retrieval pipeline (BM25 + Qdrant → Rerank)."""
    # --- Dense Retriever (Qdrant MMR) ---
    search_kwargs = {"k": 30, "fetch_k": 60}

    if target_sources:
        # Apply Qdrant metadata filter to restrict to specific sources
        source_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=src)
                )
                for src in target_sources
            ]
        )
        search_kwargs["filter"] = source_filter
        print(f"[Retriever] Filtering to sources: {target_sources}")

    qdrant_retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    # --- Sparse Retriever (BM25) ---
    try:
        if target_sources:
            filtered_docs = [d for d in all_docs if d.metadata.get("source") in target_sources]
        else:
            filtered_docs = all_docs

        if filtered_docs:
            bm25_retriever = BM25Retriever.from_documents(filtered_docs)
            bm25_retriever.k = 30

            # Reciprocal Rank Fusion (RRF) via EnsembleRetriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, qdrant_retriever],
                weights=[0.5, 0.5]
            )
        else:
            ensemble_retriever = qdrant_retriever
    except Exception as e:
        print(f"[BM25] Initialization failed, falling back to Qdrant only: {e}")
        ensemble_retriever = qdrant_retriever

    # --- Reranking (BAAI/bge-reranker-large → top 8, CACHED model) ---
    compressor = get_reranker()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    return compression_retriever
