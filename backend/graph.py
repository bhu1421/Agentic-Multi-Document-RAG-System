import time
import functools
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from backend.state import AgentState
from backend import config
from backend.llm import get_llm
from backend.vectordb import get_vector_store, get_indexed_sources
from backend.router import route_query
from backend.retrieval import (
    build_retrieval_pipeline,
    expand_to_parent_context,
    get_reranker,
    should_use_reranker,
)
from backend.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Node 1 — Router
# ──────────────────────────────────────────────
def router_node(state: AgentState):
    """Classify the query to determine the best retrieval strategy.

    Uses LLM structured output (RouterDecision Pydantic model) to return
    one of: llm_knowledge | web_search | targeted_search | all_docs.
    """
    t = time.time()
    llm = get_llm()
    indexed_sources = get_indexed_sources(state.get("user_id"))
    query = state["query"]
    route = route_query(query, llm, indexed_sources)
    elapsed = round(time.time() - t, 2)

    return {
        "strategy":       route["strategy"],
        "target_sources": route.get("sources", []),
        "source_type":    route["strategy"] if route["strategy"] in {
            "llm_knowledge", "targeted_search", "web_search"
        } else "local",
        # Accumulate timings: read existing dict, add our entry, return merged dict.
        # LangGraph replaces keys on merge, so we must preserve prior entries manually.
        "timings": {**state.get("timings", {}), "router": elapsed},
    }


def route_after_router(state: AgentState):
    """Conditional edge: decide the next node after routing."""
    strategy = state.get("strategy")
    if strategy == "llm_knowledge":
        return "generate_answer"
    if strategy == "web_search":
        return "web_search"
    return "retriever"


# ──────────────────────────────────────────────
# Node 2 — Retriever (MMR + Hierarchical + Rerank)
# ──────────────────────────────────────────────
def retriever_node(state: AgentState):
    """Execute dense retrieval, hierarchical parent expansion, and optional reranking.

    Pipeline:
    1. MMR search in Qdrant → top config.TOP_K candidate chunks
    2. Hierarchical expansion → fetch all siblings of matched parent documents
    3. Cross-encoder reranking → re-score expanded chunks, keep top config.MAX_RERANKED
    """
    t = time.time()
    store = get_vector_store()
    if not store:
        logger.info("[Retriever] Database is empty.")
        return {
            "retrieved_docs": [],
            "source_type": "llm_knowledge",
            "timings": {**state.get("timings", {}), "retriever": round(time.time() - t, 2)},
        }

    query = state["query"]
    target_sources = state.get("target_sources") or None
    source_type = state.get("source_type", "local")

    pipeline = build_retrieval_pipeline(store, target_sources, None)

    t_search = time.time()
    raw_docs = pipeline.invoke(query)
    logger.debug("[Retriever] Qdrant matched %d docs in %.1fs", len(raw_docs), time.time() - t_search)

    expanded_docs = expand_to_parent_context(raw_docs, store, user_id=state.get("user_id"))

    if not expanded_docs:
        return {
            "retrieved_docs": [],
            "source_type": source_type,
            "timings": {**state.get("timings", {}), "retriever": round(time.time() - t, 2)},
        }

    if not should_use_reranker():
        top_docs = expanded_docs[:config.MAX_RERANKED]
        logger.info("[Retriever] Reranker disabled; using top %d docs", len(top_docs))
        return {
            "retrieved_docs": top_docs,
            "source_type": source_type,
            "timings": {**state.get("timings", {}), "retriever": round(time.time() - t, 2)},
        }

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in expanded_docs]
    scores = reranker.score(pairs)
    scored_docs = sorted(zip(expanded_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in scored_docs[:config.MAX_RERANKED]]
    logger.info(
        "[Retriever] Reranked %d expanded chunks → top %d in %.1fs",
        len(expanded_docs), len(top_docs), time.time() - t,
    )

    return {
        "retrieved_docs": top_docs,
        "source_type": source_type,
        "timings": {**state.get("timings", {}), "retriever": round(time.time() - t, 2)},
    }


# ──────────────────────────────────────────────
# Node 3 — Web Search (Fallback)
# ──────────────────────────────────────────────
def web_search_node(state: AgentState):
    """Use Tavily to search the web for external or time-sensitive knowledge."""
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.documents import Document

    t = time.time()
    query = state["query"]
    search = TavilySearchResults(max_results=3)

    try:
        results = search.invoke({"query": query})
        logger.info("[WebSearch] Fetched results in %.1fs", time.time() - t)
        doc = Document(
            page_content=f"Web Search Results:\n{results}",
            metadata={"source": "tavily_web_search", "origin": "web", "file_type": "web"},
        )
        source_type = "hybrid_web" if state.get("source_type") != "web_search" else "web"
        return {
            "retrieved_docs": [doc],
            "source_type": source_type,
            "web_search_attempted": True,
            "timings": {**state.get("timings", {}), "web_search": round(time.time() - t, 2)},
        }
    except Exception as exc:
        logger.warning("[WebSearch] Search failed: %s", exc)
        source_type = "hybrid_web" if state.get("source_type") != "web_search" else "web"
        return {
            "retrieved_docs": [],
            "source_type": source_type,
            "web_search_attempted": True,
            "timings": {**state.get("timings", {}), "web_search": round(time.time() - t, 2)},
        }


# ──────────────────────────────────────────────
# Node 4 — Generate Answer
# ──────────────────────────────────────────────
def answer_node(state: AgentState):
    """Generate the final answer based on retrieved evidence.

    Hallucination guard: when using local documents, the prompt instructs the
    LLM to output exactly 'INSUFFICIENT_CONTEXT' if the retrieved chunks do not
    contain the answer.  The graph then routes to a web search fallback.
    """
    t = time.time()
    llm = get_llm()
    query = state["query"]
    chat_history = state.get("chat_history", [])
    docs = state.get("retrieved_docs", [])

    history_str = (
        "Chat History:\n"
        + "\n".join([
            f"{m['role'].capitalize()}: {m['content']}"
            for m in chat_history[-config.CHAT_HISTORY_WINDOW:]
        ])
        + "\n\n"
    ) if chat_history else ""

    # ── No documents retrieved → answer from LLM knowledge ───────────────────
    if not docs:
        current_date = datetime.now().strftime("%B %d, %Y")
        logger.info("[Answer] No docs — using LLM knowledge.")
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a friendly AI assistant. Today's date is {current_date}. "
                       f"Be helpful and concise.\n\n{{history_str}}"),
            ("human", "{input}"),
        ])
        answer = (prompt | llm).invoke({"input": query, "history_str": history_str}).content
        return {
            "answer": answer,
            "needs_web_search": False,
            "source_type": "llm_knowledge",
            "timings": {**state.get("timings", {}), "generate_answer": round(time.time() - t, 2)},
        }

    # ── Build context from retrieved documents ────────────────────────────────
    context = "\n\n".join([doc.page_content for doc in docs])
    if len(context) > config.CONTEXT_MAX_CHARS:
        context = context[:config.CONTEXT_MAX_CHARS] + "\n...[Context truncated due to length]..."

    if state.get("source_type") in {"web", "hybrid_web"}:
        system_prompt = (
            "You are an expert assistant. The context below contains web search results. "
            "Use it to answer the user's current or time-sensitive question. "
            "If the search results do not contain an exact value, say that clearly "
            "and summarise the most useful available result.\n\n"
            "{history_str}"
            "Web context:\n{context}"
        )
    else:
        system_prompt = (
            "You are an expert assistant. The context below was extracted from the user's documents. "
            "Use this context as the primary source for your answer.\n\n"
            "CRITICAL: If the context DOES NOT contain the answer, "
            "output exactly: 'INSUFFICIENT_CONTEXT'\n\n"
            "{history_str}"
            "Context:\n{context}"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    answer = (prompt | llm).invoke({
        "input": query,
        "history_str": history_str,
        "context": context,
    }).content

    # ── Insufficient context guard → trigger web search fallback ──────────────
    if "INSUFFICIENT_CONTEXT" in answer and state.get("source_type") not in {"web", "hybrid_web"}:
        logger.info("[Answer] Context insufficient. Triggering web search fallback.")
        return {
            "needs_web_search": True,
            "answer": "",
            "timings": {**state.get("timings", {}), "generate_answer": round(time.time() - t, 2)},
        }

    logger.info("[Answer] Generated in %.1fs", time.time() - t)
    return {
        "answer": answer,
        "needs_web_search": False,
        "timings": {**state.get("timings", {}), "generate_answer": round(time.time() - t, 2)},
    }


def route_after_answer(state: AgentState):
    """Conditional edge: route to web_search ONLY if context was insufficient AND web not yet tried."""
    if state.get("needs_web_search") and not state.get("web_search_attempted", False):
        return "web_search"
    return END


# ──────────────────────────────────────────────
# Graph Compilation
# ──────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def compile_graph():
    """Build and compile the LangGraph StateGraph.

    Cached with lru_cache so the (slightly expensive) compilation happens once
    per process lifetime, not once per query.

    Graph topology:
        START → router
        router  →(conditional)→ retriever | web_search | generate_answer
        retriever → generate_answer
        web_search → generate_answer
        generate_answer →(conditional)→ web_search (fallback) | END

    The cycle (generate_answer → web_search → generate_answer) is intentional:
    if local documents don't contain the answer, the graph autonomously retries
    with live web search results. The web_search_attempted flag in AgentState
    prevents infinite loops.
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("router",          router_node)
    builder.add_node("retriever",       retriever_node)
    builder.add_node("web_search",      web_search_node)
    builder.add_node("generate_answer", answer_node)

    # Entry edge
    builder.add_edge(START, "router")

    # Routing: after router, choose retriever / web_search / generate_answer
    builder.add_conditional_edges("router", route_after_router)

    # Retrieval and web search always flow into answer generation
    builder.add_edge("retriever",  "generate_answer")
    builder.add_edge("web_search", "generate_answer")

    # After answer: either done or web-search fallback (handles the cycle)
    builder.add_conditional_edges("generate_answer", route_after_answer)

    return builder.compile()
