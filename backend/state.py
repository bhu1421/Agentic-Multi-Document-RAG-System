from typing import List, TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    """The shared state passed between every node in the LangGraph workflow.

    LangGraph calls each node function with a copy of this dict and merges
    the returned partial dict back into the canonical state.  Every key here
    must be initialised before the graph is invoked (see rag._build_initial_state).
    """
    # ── Input ────────────────────────────────────────────────────────────────
    query: str
    chat_history: list
    user_id: str

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_docs: List[Document]
    target_sources: List[str]

    # ── Routing / Control ─────────────────────────────────────────────────────
    strategy: str
    source_type: str
    needs_web_search: bool
    web_search_attempted: bool   # Guard against infinite web-search loop

    # ── Output ────────────────────────────────────────────────────────────────
    answer: str

    # ── Observability ─────────────────────────────────────────────────────────
    # Each node adds its own entry: {"router": 0.4, "retriever": 1.2, ...}
    # Nodes must READ the current dict and ADD their key to avoid overwriting
    # siblings (LangGraph merges returned dicts by replacing keys, not deep-merging).
    timings: dict
