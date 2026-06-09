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

class CustomHybridRetriever:
    """A highly stable, custom implementation of Reciprocal Rank Fusion (Ensemble) + CrossEncoder Reranking."""
    def __init__(self, qdrant_retriever, bm25_retriever, reranker, top_k=8):
        self.qdrant_retriever = qdrant_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
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
            
        if not fused_docs or not self.reranker:
            return fused_docs[:self.top_k]
            
        # 4. Rerank using CrossEncoder
        pairs = [[query, doc.page_content] for doc in fused_docs]
        scores = self.reranker.score(pairs)
        
        scored_docs = list(zip(fused_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return [doc for doc, score in scored_docs[:self.top_k]]

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
    bm25_retriever = None
    try:
        if target_sources:
            filtered_docs = [d for d in all_docs if d.metadata.get("source") in target_sources]
        else:
            filtered_docs = all_docs

        if filtered_docs:
            bm25_retriever = BM25Retriever.from_documents(filtered_docs)
            bm25_retriever.k = 30
    except Exception as e:
        print(f"[BM25] Initialization failed, falling back to Qdrant only: {e}")

    # --- Reranking (BAAI/bge-reranker-large → top 8, CACHED model) ---
    reranker = get_reranker()
    
    # Return Custom Pipeline
    return CustomHybridRetriever(
        qdrant_retriever=qdrant_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        top_k=8
    )
