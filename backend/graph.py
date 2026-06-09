import time
import json
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from backend.state import AgentState
from backend.llm import get_llm
from backend.vectordb import get_vector_store, get_indexed_sources
from backend.router import route_query
from backend.retrieval import get_cached_docs, build_retrieval_pipeline

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

def retriever_node(state: AgentState):
    """Execute the retrieval pipeline for each task."""
    llm = get_llm()
    store = get_vector_store()
    tasks = state["tasks"]
    indexed_sources = get_indexed_sources()
    
    all_retrieved_docs = []
    source_type = "llm_knowledge"
    
    if not store or not indexed_sources:
        print("[Retriever] Database is empty, skipping retrieval.")
        return {"retrieved_docs": [], "source_type": "llm_knowledge"}
        
    all_docs = get_cached_docs(store)
    
    for task in tasks:
        # Route each task individually to see if it targets specific documents
        route = route_query(task, llm, indexed_sources)
        
        if route["strategy"] == "llm_knowledge":
            continue
            
        target_sources = route["sources"] if route["strategy"] == "targeted_search" else None
        source_type = "targeted_search" if route["strategy"] == "targeted_search" else "local"
        
        pipeline = build_retrieval_pipeline(store, llm, all_docs, target_sources)
        
        t = time.time()
        docs = pipeline.invoke(task)
        print(f"[Retriever] Retrieved {len(docs)} docs for task '{task}' in {time.time() - t:.1f}s")
        all_retrieved_docs.extend(docs)
        
    # Deduplicate documents based on chunk_id or page_content
    seen = set()
    unique_docs = []
    for doc in all_retrieved_docs:
        identifier = doc.metadata.get("chunk_id", doc.page_content)
        if identifier not in seen:
            seen.add(identifier)
            unique_docs.append(doc)
            
    return {"retrieved_docs": unique_docs, "source_type": source_type}

def reflection_node(state: AgentState):
    """Reflect on retrieved docs to determine if web search is needed."""
    llm = get_llm()
    query = state["query"]
    docs = state["retrieved_docs"]
    
    if not docs:
        print("[Reflection] No documents retrieved, web search needed.")
        return {"needs_web_search": True}
        
    # Format the combined context and truncate to protect LLM token limits (Groq limit is ~6000 TPM)
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

def web_search_node(state: AgentState):
    """Use DuckDuckGo to search the web for external knowledge."""
    from langchain_community.tools import DuckDuckGoSearchResults
    from langchain_core.documents import Document
    
    query = state["query"]
    search = DuckDuckGoSearchResults()
    
    t = time.time()
    try:
        results = search.invoke(query)
        print(f"[WebSearch] Fetched web results in {time.time() - t:.1f}s")
        doc = Document(page_content=f"Web Search Results:\n{results}", metadata={"source": "duckduckgo_web_search"})
        return {"retrieved_docs": state["retrieved_docs"] + [doc], "source_type": "hybrid_web"}
    except Exception as e:
        print(f"[WebSearch] Search failed: {e}")
        return {"source_type": "local"}

def route_after_reflection(state: AgentState):
    """Route to web search or answer based on reflection."""
    if state.get("needs_web_search", False):
        return "web_search"
    return "answer"

def answer_node(state: AgentState):
    """Generate the final answer based on all retrieved context."""
    llm = get_llm()
    query = state["query"]
    docs = state["retrieved_docs"]
    chat_history = state.get("chat_history", [])
    
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
        "You are an expert assistant. Use the following pieces of retrieved context to answer the question.\n"
        "You may also supplement the context with your own general knowledge to provide a more complete answer.\n\n"
        "{history_str}"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    t = time.time()
    answer = (prompt | llm).invoke({"input": query, "history_str": history_str, "context": context}).content
    print(f"[Answer] Generated final answer in {time.time() - t:.1f}s")
    
    return {"answer": answer}

# Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("reflection", reflection_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("answer", answer_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "reflection")

workflow.add_conditional_edges(
    "reflection",
    route_after_reflection,
    {"web_search": "web_search", "answer": "answer"}
)

workflow.add_edge("web_search", "answer")
workflow.add_edge("answer", END)

# Compile with MemorySaver to enable Persistence
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
