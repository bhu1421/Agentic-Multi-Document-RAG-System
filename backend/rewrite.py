"""
Query Rewriting — evaluate retrieval quality and rewrite vague queries.

Two LangGraph nodes:
    evaluate_retrieval_node()  — scores retrieval confidence
    rewrite_query_node()       — uses the LLM to produce a better search query

The evaluation → rewrite → re-retrieve cycle runs at most once
(controlled by retrieval_attempts in AgentState).
"""

import time
from langchain_core.prompts import ChatPromptTemplate
from backend import config
from backend.llm import get_llm
from backend.retrieval import should_use_reranker, get_reranker
from backend.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Node — Evaluate Retrieval Quality
# ──────────────────────────────────────────────

def evaluate_retrieval_node(state: dict) -> dict:
    """Score retrieval confidence to decide whether a query rewrite is needed.

    Confidence signals (in order of preference):
    1. Cross-encoder reranker scores (most accurate)
    2. Document count heuristic (fallback when reranker is off)

    Returns:
        retrieval_confidence: float between 0.0 and 1.0
    """
    t = time.time()
    docs = state.get("retrieved_docs", [])
    query = state.get("query", "")

    # No docs at all → zero confidence
    if not docs:
        logger.info("[EvalRetrieval] No documents retrieved → confidence 0.0")
        return {
            "retrieval_confidence": 0.0,
            "timings": {**state.get("timings", {}), "evaluate_retrieval": round(time.time() - t, 2)},
        }

    # ── Strategy 1: Cross-encoder scoring ─────────────────────────────────────
    if should_use_reranker():
        try:
            reranker = get_reranker()
            # Score top-N docs for speed (no need to score all)
            eval_docs = docs[:5]
            pairs = [[query, doc.page_content] for doc in eval_docs]
            scores = reranker.score(pairs)

            # Normalise: cross-encoder scores are typically in [-10, 10] range
            # Map to [0, 1] using sigmoid-like scaling
            import math
            normalised = [1 / (1 + math.exp(-s)) for s in scores]
            avg_score = sum(normalised) / len(normalised)

            logger.info(
                "[EvalRetrieval] Cross-encoder avg=%.3f (raw: %s) | docs=%d",
                avg_score, [round(s, 2) for s in scores[:3]], len(docs),
            )
            return {
                "retrieval_confidence": round(avg_score, 4),
                "timings": {**state.get("timings", {}), "evaluate_retrieval": round(time.time() - t, 2)},
            }
        except Exception as exc:
            logger.warning("[EvalRetrieval] Reranker scoring failed: %s", exc)

    # ── Strategy 2: Count-based heuristic ─────────────────────────────────────
    # Simple but surprisingly effective: if we got a decent number of docs
    # with meaningful content, the retrieval is probably adequate.
    non_empty = [d for d in docs if len(d.page_content.strip()) > 50]
    if len(non_empty) >= 3:
        confidence = min(1.0, len(non_empty) / 8.0)  # 8+ docs → 1.0
    elif len(non_empty) >= 1:
        confidence = 0.25
    else:
        confidence = 0.0

    logger.info(
        "[EvalRetrieval] Heuristic confidence=%.2f (non-empty docs=%d/%d)",
        confidence, len(non_empty), len(docs),
    )
    return {
        "retrieval_confidence": round(confidence, 4),
        "timings": {**state.get("timings", {}), "evaluate_retrieval": round(time.time() - t, 2)},
    }


# ──────────────────────────────────────────────
# Node — Rewrite Query
# ──────────────────────────────────────────────

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a search query optimiser for a document retrieval system. "
     "Your job is to rewrite the user's query so it retrieves better results.\n\n"
     "Rules:\n"
     "1. Preserve the original intent — do NOT change the meaning.\n"
     "2. Make the query more specific and descriptive.\n"
     "3. Expand abbreviations and ambiguous terms.\n"
     "4. Add relevant context keywords that improve retrieval.\n"
     "5. Return ONLY the rewritten query — no explanations, no quotes."),
    ("human", "Original query: {query}\n\nRewritten query:"),
])


def rewrite_query_node(state: dict) -> dict:
    """Use the LLM to rewrite a vague query for better retrieval.

    Preserves the original query in state. Increments retrieval_attempts
    to prevent infinite loops. Clears retrieved_docs so the retriever
    fetches fresh results with the rewritten query.
    """
    t = time.time()
    original_query = state.get("original_query") or state.get("query", "")
    current_query = state.get("query", "")
    attempts = state.get("retrieval_attempts", 0)

    llm = get_llm()

    try:
        rewritten = (
            _REWRITE_PROMPT | llm
        ).invoke({"query": current_query}).content.strip()

        # Sanity check: if the LLM returns something empty or identical, keep original
        if not rewritten or rewritten.lower() == current_query.lower():
            logger.info("[Rewrite] LLM returned identical/empty rewrite — keeping original")
            rewritten = current_query
        else:
            logger.info(
                "[Rewrite] '%s' → '%s' (attempt %d)",
                current_query, rewritten, attempts + 1,
            )
    except Exception as exc:
        logger.warning("[Rewrite] LLM rewrite failed: %s — keeping original", exc)
        rewritten = current_query

    return {
        "query": rewritten,
        "original_query": original_query,
        "rewritten_query": rewritten,
        "retrieval_attempts": attempts + 1,
        "retrieved_docs": [],  # Clear so retriever fetches fresh results
        "timings": {**state.get("timings", {}), "rewrite_query": round(time.time() - t, 2)},
    }


# ──────────────────────────────────────────────
# Conditional Edge — Route After Evaluation
# ──────────────────────────────────────────────

def route_after_evaluation(state: dict) -> str:
    """Decide whether to rewrite the query or proceed to answer generation.

    Logic:
    1. If confidence >= threshold → generate_answer
    2. If already retried (attempts >= max) → generate_answer
    3. Otherwise → rewrite_query
    """
    confidence = state.get("retrieval_confidence", 1.0)
    attempts = state.get("retrieval_attempts", 0)
    threshold = config.RETRIEVAL_CONFIDENCE_THRESHOLD
    max_attempts = config.MAX_RETRIEVAL_ATTEMPTS

    if confidence >= threshold:
        logger.info(
            "[EvalRoute] Confidence %.3f >= %.3f → generate_answer",
            confidence, threshold,
        )
        return "generate_answer"

    if attempts >= max_attempts:
        logger.info(
            "[EvalRoute] Already retried %d/%d → generate_answer (despite low confidence %.3f)",
            attempts, max_attempts, confidence,
        )
        return "generate_answer"

    logger.info(
        "[EvalRoute] Low confidence %.3f < %.3f (attempt %d/%d) → rewrite_query",
        confidence, threshold, attempts, max_attempts,
    )
    return "rewrite_query"
