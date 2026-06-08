import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

# Local persistent Qdrant
QDRANT_PATH = "local_qdrant"

def get_vector_store():
    """Get or create the Qdrant vector store using BGE-M3."""
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )
    
    # If the path doesn't exist, we must create a new store later
    if not os.path.exists(QDRANT_PATH):
        return None
        
    try:
        store = Qdrant.from_existing_collection(
            embedding=embeddings,
            collection_name="rag_collection",
            path=QDRANT_PATH
        )
        return store
    except Exception:
        return None

def store_documents(chunks):
    """Store document chunks in Qdrant."""
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )
    
    store = Qdrant.from_documents(
        chunks,
        embeddings,
        path=QDRANT_PATH,
        collection_name="rag_collection",
        force_recreate=True
    )
    return store
