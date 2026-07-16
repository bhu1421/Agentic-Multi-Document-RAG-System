import time
from typing import Generator
from backend.graph import compile_graph
from backend.logger import get_logger

logger = get_logger(__name__)

# Human-readable labels for each LangGraph node shown in the streaming UI
NODE_LABELS: dict[str, str] = {
    "cache_check":        "⚡ Checking cache...",
    "guardrail":          "🛡️ Checking guardrails...",
    "router":             "🔀 Routing query...",
    "retriever":          "📂 Retrieving documents...",
    "evaluate_retrieval": "📊 Evaluating retrieval...",
    "rewrite_query":      "✏️ Rewriting query...",
    "web_search":         "🌐 Searching the web...",
    "generate_answer":    "✍️  Generating answer...",
    "cache_store":        "💾 Caching result...",
}


def _build_initial_state(query: str, chat_history: list, user_id: str = "public") -> dict:
    """Construct a fully-initialised AgentState dict.

    Every key defined in AgentState must be present here.
    LangGraph will raise a KeyError for any uninitialised key accessed by a node.
    """
    return {
        "query":                query,
        "chat_history":         chat_history or [],
        "user_id":              user_id,
        "guardrail_result":     "",
        "guardrail_reason":     "",
        "retrieved_docs":       [],
        "target_sources":       [],
        "original_query":       query,
        "rewritten_query":      "",
        "retrieval_attempts":   0,
        "retrieval_confidence": 0.0,
        "answer":               "",
        "strategy":             "",
        "source_type":          "",
        "needs_web_search":     False,
        "web_search_attempted": False,
        "cache_hit":            False,
        "timings":              {},   # Populated incrementally by each node
    }


def stream_agentic_response(
    query: str,
    chat_history: list = None,
    session_id: str = "default",
    user_id: str = "public",
) -> Generator[tuple[str, object], None, None]:
    """Stream the agentic RAG pipeline, yielding progress events and the final answer.

    Uses LangGraph's compiled StateGraph and its native .stream() method
    (stream_mode='updates'), which yields one dict per node as it completes:
        {"node_name": {state_delta}}

    Yields tuples of (event_type, data):
        ("node",   node_name: str)    — fired after each graph node completes
        ("answer", answer_text: str)  — the final generated answer
        ("meta",   meta_dict: dict)   — source_type, docs, elapsed, timings
        ("error",  message: str)      — if the pipeline crashes
    """
    total_start = time.time()
    initial_state = _build_initial_state(query, chat_history, user_id)
    final = initial_state.copy()
    graph = compile_graph()

    try:
        # graph.stream with stream_mode="updates" yields:
        #   {"node_name": {partial_state_update}} for each completed node.
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            node_name = next(iter(chunk))
            updates = chunk[node_name]
            final.update(updates)
            yield ("node", node_name)

    except Exception:
        logger.exception("[Stream] Pipeline failed for session '%s'", session_id)
        yield ("error", "The pipeline encountered an error. Please try again.")
        return

    answer = final.get("answer") or "I couldn't generate an answer."
    source_type = final.get("source_type", "local")
    docs = final.get("retrieved_docs") or []
    timings = final.get("timings", {})
    elapsed = time.time() - total_start

    logger.info(
        "[Stream] session=%s  %.1fs  source=%s  docs=%d  timings=%s",
        session_id, elapsed, source_type, len(docs), timings,
    )

    yield ("answer", answer)
    yield ("meta", {
        "source_type": source_type,
        "docs":        docs,
        "elapsed":     elapsed,
        "timings":     timings,
    })


def get_agentic_response(
    query: str,
    chat_history: list = None,
    session_id: str = "default",
    user_id: str = "public",
) -> dict | None:
    """Non-streaming entry point — collects the full pipeline result synchronously.

    Uses LangGraph's .invoke() which runs the graph to completion and returns
    the final state dict.  Prefer stream_agentic_response() for the UI.
    """
    total_start = time.time()
    initial_state = _build_initial_state(query, chat_history, user_id)
    graph = compile_graph()

    try:
        final_state = graph.invoke(initial_state)
    except Exception:
        logger.exception("[Graph] Pipeline failed for session '%s'", session_id)
        return None

    answer = final_state.get("answer", "I couldn't generate an answer.")
    source_type = final_state.get("source_type", "local")
    docs = final_state.get("retrieved_docs", [])
    timings = final_state.get("timings", {})

    logger.info(
        "[Total] session=%s  %.1fs  source=%s  docs=%d",
        session_id, time.time() - total_start, source_type, len(docs),
    )

    return {"answer": answer, "context": docs, "source_type": source_type, "timings": timings}
