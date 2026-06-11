import time
import json
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from backend.state import AgentState
from backend.llm import get_llm
from backend.vectordb import get_vector_store, get_indexed_sources
from backend.router import route_query
from backend.retrieval import get_cached_docs, get_reranker, build_retrieval_pipeline

# ──────────────────────────────────────────────
# Node 0 — Router (NEW)
# ──────────────────────────────────────────────
def router_node(state: AgentState):
    """Classify the query to determine if we need retrieval or can answer directly."""
    llm = get_llm()
    indexed_sources = get_indexed_sources()
    route = route_query(state["query"], llm, indexed_sources)
    return {
        "strategy": route["strategy"],
        "target_sources": route.get("sources", []),
        "source_type": route["strategy"] if route["strategy"] == "targeted_search" else "local"
    }

def route_after_router(state: AgentState):
    """Route directly to answer if LLM knowledge is sufficient, else planner."""
    if state.get("strategy") == "llm_knowledge":
        return "generate_answer"
    return "planner"

# ──────────────────────────────────────────────
# Node 1 — Planner
# ──────────────────────────────────────────────
def planner_node(state: AgentState):
    """Determine the retrieval tasks needed to answer the query."""
    llm = get_llm()
    query = state["query"]
    chat_history = state.get("chat_history", [])
    
    # Rewrite the query to be context-aware using chat history
    from backend.memory import rewrite_query
    if chat_history:
        query = rewrite_query(query, chat_history, llm)
    
    # Simple Planner Prompt
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a planner for a RAG system. Your job is to break down the user's query "
         "into a JSON list of independent retrieval tasks. \n\n"
         "For simple queries, output a list with one task. For complex queries (e.g. comparing two things), "
         "output multiple tasks.\n\n"
         "Output ONLY valid JSON representing a list of strings, e.g., [\"Task 1\", \"Task 2\"]. "
         "Do not include markdown blocks or any other text."),
        ("human", "Query: {query}")
    ])
    
    t = time.time()
    response = (planner_prompt | llm).invoke({"query": query}).content.strip()
    print(f"[Planner] Generated tasks in {time.time() - t:.1f}s: {response}")
    
    try:
        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "").strip()
        tasks = json.loads(response)
        if not isinstance(tasks, list):
            tasks = [query]
    except Exception as e:
        print(f"[Planner] Failed to parse JSON, falling back to original query. Error: {e}")
        tasks = [query]
        
    return {"query": query, "tasks": tasks}


# ──────────────────────────────────────────────
# Node 2 — Query Expansion  (NEW)
# ──────────────────────────────────────────────
from backend.agents.query_expansion import query_expansion_node


# ──────────────────────────────────────────────
# Node 3 — Metadata Filter  (NEW)
# ──────────────────────────────────────────────
from backend.agents.metadata_filter import metadata_filter_node


# ──────────────────────────────────────────────
# Node 4 — Retriever  (UPDATED)
# ──────────────────────────────────────────────
def retriever_node(state: AgentState):
    """Execute the retrieval pipeline using expanded queries and metadata filters."""
    llm = get_llm()
    store = get_vector_store()
    indexed_sources = get_indexed_sources()

    # Use expanded queries if available, otherwise fall back to tasks or raw query
    expanded_queries = state.get("expanded_queries") or state.get("tasks") or [state["query"]]
    metadata_filters = state.get("metadata_filters") or {}
    
    if not store or not indexed_sources:
        print("[Retriever] Database is empty, skipping retrieval.")
        return {"retrieved_docs": [], "source_type": "llm_knowledge"}
    
    all_docs = get_cached_docs(store)
    
    # Strategy is already determined by router_node
    strategy = state.get("strategy", "all_docs")
    if strategy == "llm_knowledge":
        return {"retrieved_docs": [], "source_type": "llm_knowledge"}
    
    # Start with router's source targets
    target_sources = state.get("target_sources")
    source_type = state.get("source_type", "local")
    
    # If Metadata Filter extracted a source, validate and use it
    if "source" in metadata_filters:
        meta_source = metadata_filters["source"]
        indexed_lower = {s.lower(): s for s in indexed_sources}
        if meta_source.lower() in indexed_lower:
            target_sources = [indexed_lower[meta_source.lower()]]
            source_type = "targeted_search"
            print(f"[Retriever] Metadata filter overriding source to: {target_sources}")

    # Build the pipeline ONCE with combined filters
    pipeline = build_retrieval_pipeline(store, llm, all_docs, target_sources, metadata_filters)
    
    # Search with EVERY expanded query to maximise recall
    all_retrieved_docs = []
    for eq in expanded_queries:
        t = time.time()
        docs = pipeline.invoke(eq)
        print(f"[Retriever] Retrieved {len(docs)} docs for '{eq}' ({time.time() - t:.1f}s)")
        all_retrieved_docs.extend(docs)
        
    # Deduplicate documents based on chunk_id or content prefix
    seen = set()
    unique_docs = []
    for doc in all_retrieved_docs:
        identifier = doc.metadata.get("chunk_id", doc.page_content[:200])
        if identifier not in seen:
            seen.add(identifier)
            unique_docs.append(doc)

    print(f"[Retriever] Total: {len(all_retrieved_docs)} raw -> {len(unique_docs)} unique docs")
    return {"retrieved_docs": unique_docs, "source_type": source_type}


# ──────────────────────────────────────────────
# Node 5 — Reflection
# ──────────────────────────────────────────────
def reflection_node(state: AgentState):
    """Reflect on retrieved docs to determine if web search is needed."""
    llm = get_llm()
    query = state["query"]
    docs = state["retrieved_docs"]
    
    # If router decided this is an LLM-knowledge query (e.g. greetings),
    # don't trigger web search — just go straight to answer.
    if state.get("source_type") == "llm_knowledge":
        print("[Reflection] LLM knowledge query, skipping web search.")
        return {"needs_web_search": False}
    
    if not docs:
        print("[Reflection] No documents retrieved, web search needed.")
        return {"needs_web_search": True}
        
    # Format the combined context and truncate to protect LLM token limits
    context = "\n\n".join([doc.page_content for doc in docs])
    if len(context) > 12000:
        context = context[:12000] + "\n...[Context truncated due to length]..."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a critical evaluator for a RAG system. Your job is to determine if the retrieved context "
         "contains sufficient evidence to answer the user's query.\n\n"
         "If the context contains the answer, output 'YES'.\n"
         "If the context is completely irrelevant or missing key information to answer the query, output 'NO'.\n"
         "Output ONLY 'YES' or 'NO'. Do not output any other text."),
        ("human", "Query: {query}\n\nContext:\n{context}")
    ])
    
    response = (prompt | llm).invoke({"query": query, "context": context}).content.strip().upper()
    needs_search = "NO" in response
    
    print(f"[Reflection] Evidence sufficient? {'No (Needs Web Search)' if needs_search else 'Yes'}")
    return {"needs_web_search": needs_search}


# ──────────────────────────────────────────────
# Node 6 — Web Search
# ──────────────────────────────────────────────
def web_search_node(state: AgentState):
    """Use DuckDuckGo to search the web for external knowledge.

    Results are stored in `web_docs` (separate from `retrieved_docs`)
    so that Evidence Fusion can tag provenance before merging.
    """
    from langchain_community.tools import DuckDuckGoSearchResults
    from langchain_core.documents import Document
    
    query = state["query"]
    search = DuckDuckGoSearchResults()
    
    t = time.time()
    try:
        results = search.invoke(query)
        print(f"[WebSearch] Fetched web results in {time.time() - t:.1f}s")
        doc = Document(
            page_content=f"Web Search Results:\n{results}",
            metadata={
                "source": "duckduckgo_web_search",
                "origin": "web",
                "file_type": "web"
            }
        )
        return {"web_docs": [doc], "source_type": "hybrid_web"}
    except Exception as e:
        print(f"[WebSearch] Search failed: {e}")
        return {"web_docs": []}


# ──────────────────────────────────────────────
# Node 7 — Evidence Fusion  (NEW)
# ──────────────────────────────────────────────
from backend.agents.evidence_fusion import evidence_fusion_node


# ──────────────────────────────────────────────
# Node 8 — Reranker  (EXTRACTED from retrieval.py)
# ──────────────────────────────────────────────
def reranker_node(state: AgentState):
    """Rerank all fused evidence using the BAAI cross-encoder.

    Operates on the full fused evidence (local + web + metadata-filtered)
    instead of only local results, producing a final top-8 ranking.
    """
    docs = state.get("fused_docs", [])
    query = state["query"]
    
    if not docs:
        print("[Reranker] No documents to rerank")
        return {"reranked_docs": []}
    
    reranker = get_reranker()
    
    t = time.time()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.score(pairs)
    
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    top_docs = [doc for doc, score in scored_docs[:8]]
    
    print(f"[Reranker] Reranked {len(docs)} docs -> top {len(top_docs)} ({time.time() - t:.1f}s)")
    return {"reranked_docs": top_docs}


# ──────────────────────────────────────────────
# Node 9 — Answer
# ──────────────────────────────────────────────
def answer_node(state: AgentState):
    """Generate the final answer based on reranked evidence."""
    llm = get_llm()
    query = state["query"]
    chat_history = state.get("chat_history", [])

    # Use reranked docs first; fall back to fused, then retrieved
    docs = (
        state.get("reranked_docs")
        or state.get("fused_docs")
        or state.get("retrieved_docs")
        or []
    )
    
    if not docs:
        print("[Answer] No documents retrieved, falling back to LLM knowledge.")
        history_str = "Chat History:\n" + "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-10:]]) + "\n\n" if chat_history else ""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a friendly AI assistant. Answer the user's question using your own general knowledge. "
             "Be helpful, concise, and conversational.\n\n{history_str}"),
            ("human", "{input}"),
        ])
        answer = (prompt | llm).invoke({"input": query, "history_str": history_str}).content
        return {"answer": answer, "source_type": "llm_knowledge"}
        
    # Format the combined context and truncate to protect LLM token limits
    context = "\n\n".join([doc.page_content for doc in docs])
    if len(context) > 12000:
        context = context[:12000] + "\n...[Context truncated due to length]..."
    
    history_str = "Chat History:\n" + "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-10:]]) + "\n\n" if chat_history else ""
    
    system_prompt = (
        "You are an expert assistant. The context below was extracted directly from the user's own "
        "uploaded documents (PDFs, text files, code, web pages, etc.). When the user refers to "
        "'my resume', 'the document', 'the file', or 'my data', they are referring to THIS context — "
        "it IS their document content.\n\n"
        "Use this context as the primary source for your answer. You may supplement with your own "
        "general knowledge only when the context is insufficient.\n\n"
        "{history_str}"
        "Context from user's documents:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    t = time.time()
    answer = (prompt | llm).invoke({"input": query, "history_str": history_str, "context": context}).content
    print(f"[Answer] Generated final answer in {time.time() - t:.1f}s")
    
    return {"answer": answer}


# ──────────────────────────────────────────────
# Conditional Edge Router
# ──────────────────────────────────────────────
def route_after_reflection(state: AgentState):
    """Route to web search or directly to evidence fusion."""
    if state.get("needs_web_search", False):
        return "web_search"
    return "evidence_fusion"


# ══════════════════════════════════════════════
# Build the Graph
# ══════════════════════════════════════════════
#
# Flow:
#   START → planner → query_expansion → metadata_filter → retriever
#         → reflection → [web_search?] → evidence_fusion → reranker → answer → END
#
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_expansion", query_expansion_node)
workflow.add_node("metadata_filter", metadata_filter_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("reflection", reflection_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("evidence_fusion", evidence_fusion_node)
workflow.add_node("reranker", reranker_node)
workflow.add_node("generate_answer", answer_node)

# Linear edges
workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    route_after_router,
    {"planner": "planner", "generate_answer": "generate_answer"}
)
workflow.add_edge("planner", "query_expansion")
workflow.add_edge("query_expansion", "metadata_filter")
workflow.add_edge("metadata_filter", "retriever")
workflow.add_edge("retriever", "reflection")

# Conditional: reflection decides if web search is needed
workflow.add_conditional_edges(
    "reflection",
    route_after_reflection,
    {"web_search": "web_search", "evidence_fusion": "evidence_fusion"}
)

# Both paths converge at evidence_fusion
workflow.add_edge("web_search", "evidence_fusion")
workflow.add_edge("evidence_fusion", "reranker")
workflow.add_edge("reranker", "generate_answer")
workflow.add_edge("generate_answer", END)

# Compile with MemorySaver to enable Persistence
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
