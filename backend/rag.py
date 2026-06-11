import time
from backend.graph import app
from langchain_core.documents import Document

def get_agentic_response(query: str, chat_history: list = None):
    """Orchestrates the full agentic RAG pipeline using LangGraph."""
    total_start = time.time()
    
    # Run the graph
    initial_state = {
        "query": query,
        "chat_history": chat_history or [],
        "tasks": [],
        "expanded_queries": [],
        "metadata_filters": {},
        "retrieved_docs": [],
        "web_docs": [],
        "fused_docs": [],
        "reranked_docs": [],
        "answer": "",
        "strategy": "",
        "target_sources": [],
        "source_type": "",
        "needs_web_search": False
    }
    
    try:
        config = {"configurable": {"thread_id": "default_user_session"}}
        final_state = app.invoke(initial_state, config=config)
    except Exception as e:
        print(f"[Graph Error] {e}")
        return None
        
    answer = final_state.get("answer", "I couldn't generate an answer.")
    source_type = final_state.get("source_type", "local")

    # Use reranked docs for source citations; fall back gracefully
    docs = (
        final_state.get("reranked_docs")
        or final_state.get("fused_docs")
        or final_state.get("retrieved_docs")
        or []
    )
    
    print(f"[Total] Completed in {time.time() - total_start:.1f}s ({source_type})")
    
    return {
        "answer": answer,
        "context": docs,
        "source_type": source_type
    }
