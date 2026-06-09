import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Local persistent Qdrant
QDRANT_PATH = "local_qdrant"

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Force CPU to prevent meta tensor bugs
    )

@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    from qdrant_client import QdrantClient
    os.makedirs(QDRANT_PATH, exist_ok=True)
    return QdrantClient(path=QDRANT_PATH)

def get_vector_store():
    """Get or create the Qdrant vector store."""
    embeddings = get_embeddings()
    client = get_qdrant_client()
    
    try:
        client.get_collection("rag_collection")
    except Exception:
        from qdrant_client.http.models import VectorParams, Distance
        client.create_collection(
            collection_name="rag_collection",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        
    return QdrantVectorStore(client=client, collection_name="rag_collection", embedding=embeddings)

def store_documents(chunks):
    """Store document chunks in Qdrant."""
    store = get_vector_store()
    store.add_documents(chunks)
    return store

def get_indexed_sources():
    """Retrieve all unique 'source' metadata values from Qdrant."""
    store = get_vector_store()
    if not store:
        return []
    
    try:
        # Scroll through points to grab metadata
        records, _ = store.client.scroll(
            collection_name="rag_collection",
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        sources = set()
        for record in records:
            # Langchain stores metadata inside the 'metadata' payload key
            metadata = record.payload.get("metadata", {})
            source = metadata.get("source")
            if source:
                if str(source).startswith("http"):
                    clean_source = source
                else:
                    # Clean up the path to just the filename (e.g., uploaded_docs\resume.pdf -> resume.pdf)
                    clean_source = os.path.basename(source)
                if clean_source:
                    sources.add(clean_source)
        return sorted(list(sources))
    except Exception as e:
        print(f"Error in get_indexed_sources: {e}")
        return []

def delete_source(source_name: str):
    """Delete all chunks belonging to a specific source."""
    from qdrant_client.http import models
    
    store = get_vector_store()
    if not store:
        return False
        
    try:
        store.client.delete(
            collection_name="rag_collection",
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=source_name)
                    )
                ]
            )
        )
        return True
    except Exception:
        return False
