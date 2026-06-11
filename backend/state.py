from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
import operator

class AgentState(TypedDict):
    """The shared state for the LangGraph workflow."""
    query: str
    chat_history: list
    
    # Planner output
    tasks: List[str]
    
    # Query Expansion output
    expanded_queries: List[str]
    
    # Metadata Filter output
    metadata_filters: dict
    
    # Retriever output.
    retrieved_docs: List[Document]
    
    # Web Search results (kept separate until Evidence Fusion merges them)
    web_docs: List[Document]
    
    # Evidence Fusion output
    fused_docs: List[Document]
    
    # Reranker output (final ranked evidence for the Answer agent)
    reranked_docs: List[Document]
    
    # Router output
    strategy: str
    target_sources: List[str]
    
    # Final Answer output
    answer: str
    source_type: str
    needs_web_search: bool
