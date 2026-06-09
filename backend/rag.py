import time
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Import from modularized backend components
from backend.llm import get_llm
from backend.vectordb import get_vector_store, get_indexed_sources
from backend.router import route_query
from backend.retrieval import get_cached_docs, build_retrieval_pipeline

def _get_llm_response(query: str, llm) -> dict:
    """Answer directly from the LLM's own knowledge."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a friendly and knowledgeable AI assistant. "
         "Answer the user's question using your own general knowledge. "
         "Be helpful, concise, and conversational."),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    answer = chain.invoke({"input": query})
    return {
        "answer": answer.content,
        "context": [Document(
            page_content="Response generated from LLM general knowledge.",
            metadata={"source": "LLM General Knowledge", "file_type": "llm_knowledge"}
        )],
        "source_type": "llm_knowledge"
    }

def get_agentic_response(query: str):
    """Orchestrates the full agentic RAG pipeline with intelligent query routing."""
    total_start = time.time()
    
    store = get_vector_store()
    llm = get_llm()

    # ── No documents indexed at all → LLM knowledge only ──
    if not store:
        return _get_llm_response(query, llm)

    # ── Get indexed sources for routing decisions ──
    indexed_sources = get_indexed_sources()

    # ── No documents in collection → LLM knowledge only ──
    if not indexed_sources:
        print("[Router] No indexed sources found, using LLM knowledge")
        result = _get_llm_response(query, llm)
        print(f"[Total] Completed in {time.time() - total_start:.1f}s (LLM knowledge — no docs)")
        return result

    # ── Route the query ──
    route = route_query(query, llm, indexed_sources)

    # ── Strategy 1: LLM Knowledge (skip retrieval entirely) ──
    if route["strategy"] == "llm_knowledge":
        result = _get_llm_response(query, llm)
        print(f"[Total] Completed in {time.time() - total_start:.1f}s (LLM knowledge)")
        return result

    # ── Load all docs from cache for BM25 index ──
    all_docs = get_cached_docs(store)

    # ── Strategy 2 & 3: Document Search (targeted or all) ──
    target_sources = route["sources"] if route["strategy"] == "targeted_search" else None

    compression_retriever = build_retrieval_pipeline(store, llm, all_docs, target_sources)

    # RAG prompt — allows hybrid answers (docs + LLM knowledge)
    system_prompt = (
        "You are an expert assistant. Use the following pieces of retrieved context to answer the question.\n"
        "You may also supplement the context with your own general knowledge to provide a more complete answer.\n"
        "If the context is completely irrelevant and you cannot answer at all, reply exactly with: WEB_SEARCH_REQUIRED\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(compression_retriever, qa_chain)

    local_response = rag_chain.invoke({"input": query})
    answer = local_response["answer"].strip()

    # ── Check for refusal → Wikipedia fallback ──
    refusal_phrases = [
        "unable to find", "don't know", "not mention",
        "not provided", "unable to locate", "WEB_SEARCH_REQUIRED"
    ]
    needs_fallback = any(phrase.lower() in answer.lower() for phrase in refusal_phrases)

    if needs_fallback:
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
        search_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

        try:
            search_results = search_tool.invoke(query)
            if not search_results or "No good Wikipedia Search Result" in search_results:
                search_results = "No results found on Wikipedia."
        except Exception as e:
            search_results = f"Web search failed: {e}"

        web_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert assistant. Answer the user's question using the "
             "following Wikipedia search results:\n\n{search_results}"),
            ("human", "{input}")
        ])

        web_chain = web_prompt | llm
        final_answer = web_chain.invoke({"search_results": search_results, "input": query})

        print(f"[Total] Completed in {time.time() - total_start:.1f}s (Wikipedia fallback)")
        return {
            "answer": final_answer.content,
            "context": [Document(
                page_content=search_results,
                metadata={"source": "Wikipedia Agentic Search API", "file_type": "web_search"}
            )],
            "source_type": "web"
        }

    # ── Success — return document-based answer ──
    source_type = "targeted_search" if route["strategy"] == "targeted_search" else "local"

    print(f"[Total] Completed in {time.time() - total_start:.1f}s ({source_type})")
    return {
        "answer": answer,
        "context": local_response["context"],
        "source_type": source_type
    }
