from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
import operator

class AgentState(TypedDict):
    """The shared state for the LangGraph workflow."""
    query: str
    chat_history: list
    
    # Planner output
    tasks: List[str]
    
    # Retriever output. We use `operator.add` so that multiple tasks can append documents to the same state list concurrently if needed.
    retrieved_docs: Annotated[List[Document], operator.add]
    
    # Router output
    strategy: str
    target_sources: List[str]
    
    # Final Answer output
    answer: str
    source_type: str
    needs_web_search: bool
