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
        "retrieved_docs": [],
        "answer": "",
        "strategy": "",
        "target_sources": [],
        "source_type": ""
    }
    
    try:
        config = {"configurable": {"thread_id": "default_user_session"}}
        final_state = app.invoke(initial_state, config=config)
    except Exception as e:
        print(f"[Graph Error] {e}")
        return None
        
    answer = final_state.get("answer", "I couldn't generate an answer.")
    source_type = final_state.get("source_type", "local")
    docs = final_state.get("retrieved_docs", [])
    
    print(f"[Total] Completed in {time.time() - total_start:.1f}s ({source_type})")
    
    return {
        "answer": answer,
        "context": docs,
        "source_type": source_type
    }
