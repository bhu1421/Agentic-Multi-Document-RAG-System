import time
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from backend.logger import get_logger

logger = get_logger(__name__)


# ── Structured Output Schema ───────────────────────────────────────────────────

class RouterDecision(BaseModel):
    """Structured routing decision produced by the LLM.

    Using Pydantic + LLM structured output instead of regex string-matching
    eliminates an entire class of parsing bugs and makes the router's output
    directly inspectable and type-safe.
    """
    strategy: Literal["llm_knowledge", "web_search", "targeted_search", "all_docs"] = Field(
        description=(
            "llm_knowledge — greetings or general knowledge with no document relevance. "
            "web_search — live/current/today/price/weather/news questions. "
            "targeted_search — query clearly about a specific named document. "
            "all_docs — broad questions about uploaded documents or when unsure."
        )
    )
    sources: List[str] = Field(
        default_factory=list,
        description="For targeted_search only: exact filenames from the available sources list.",
    )
    reasoning: str = Field(
        default="",
        description="One-sentence justification for the routing decision (used for logging).",
    )


# ── Router Function ────────────────────────────────────────────────────────────

def route_query(query: str, llm, indexed_sources: list) -> dict:
    """Use the LLM to classify the query into a retrieval strategy.

    Returns:
        dict with keys:
            - strategy: 'llm_knowledge' | 'web_search' | 'targeted_search' | 'all_docs'
            - sources:  list of source names (only for targeted_search)
    """
    sources_list = (
        "\n".join(f"- {s}" for s in indexed_sources)
        if indexed_sources else "- (no documents indexed)"
    )

    # ── Fast-path: hardcoded bypass for unambiguous document-action queries ────
    query_lower = query.lower()
    if any(action in query_lower for action in ["analyse", "analyze", "summarize", "read"]) and \
       any(target in query_lower for target in ["resume", "document", "file", "uploaded"]):
        logger.info("[Router] Query: '%s' -> Hardcoded bypass: ALL_DOCS", query)
        return {"strategy": "all_docs", "sources": []}

    # ── LLM-based structured routing ──────────────────────────────────────────
    router_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a query router for a RAG (Retrieval-Augmented Generation) system.\n\n"
         "Available indexed documents:\n"
         f"{sources_list}\n\n"
         "Routing strategies:\n"
         "1. llm_knowledge — Use for greetings, casual chat, or general knowledge "
         "questions that have NOTHING to do with any documents.\n"
         "2. web_search — Use for current/live/today/latest/price/rate/weather/news "
         "questions that require up-to-date information.\n"
         "3. targeted_search — Use when the query CLEARLY references a specific named "
         "document from the available list above. Put the exact filename(s) in 'sources'.\n"
         "4. all_docs — Use for broad questions about 'my documents', 'the uploaded file', "
         "'the resume', or any research question spanning all documents.\n\n"
         "CRITICAL RULES:\n"
         "- If the user asks to analyze/read/summarize 'the document' or 'my file', "
         "choose all_docs or targeted_search. NEVER llm_knowledge.\n"
         "- If the query contains 'today', 'current', 'live', 'price', 'rate', or 'news', "
         "choose web_search. NEVER llm_knowledge.\n"
         "- If no documents are indexed, prefer llm_knowledge for document-type queries."),
        ("human", "{query}"),
    ])

    structured_llm = llm.with_structured_output(RouterDecision)
    router_chain = router_prompt | structured_llm

    t = time.time()
    try:
        raw = router_chain.invoke({"query": query})
        # with_structured_output returns a Pydantic model on newer langchain-groq
        # and a plain dict on older versions (e.g. 0.1.5). Handle both.
        if isinstance(raw, dict):
            decision = RouterDecision(**raw)
        else:
            decision = raw
    except Exception as exc:
        logger.warning(
            "[Router] Structured output failed: %s. Defaulting to all_docs.", exc
        )
        return {"strategy": "all_docs", "sources": []}

    logger.info(
        "[Router] Query: '%s' -> Strategy: %s | Sources: %s | Reason: '%s' (%.1fs)",
        query, decision.strategy, decision.sources, decision.reasoning, time.time() - t,
    )

    # ── Validate targeted sources against what is actually indexed ─────────────
    if decision.strategy == "targeted_search":
        indexed_lower = {s.lower(): s for s in indexed_sources}
        valid_sources = [
            indexed_lower[s.lower()]
            for s in decision.sources
            if s.lower() in indexed_lower
        ]
        if valid_sources:
            return {"strategy": "targeted_search", "sources": valid_sources}
        else:
            logger.warning(
                "[Router] Sources %s not found in index. Falling back to all_docs.",
                decision.sources,
            )
            return {"strategy": "all_docs", "sources": []}

    return {"strategy": decision.strategy, "sources": decision.sources}
