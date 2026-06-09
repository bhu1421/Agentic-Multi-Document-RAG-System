import time
from langchain_core.prompts import ChatPromptTemplate

def route_query(query: str, llm, indexed_sources: list) -> dict:
    """Use the LLM to classify the query into a retrieval strategy.
    
    Returns:
        dict with keys:
            - strategy: 'llm_knowledge' | 'targeted_search' | 'all_docs'
            - sources:  list of source names (only for targeted_search)
    """
    sources_list = "\n".join(f"- {s}" for s in indexed_sources) if indexed_sources else "- (none)"

    router_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a query router for a RAG system. Classify the user's query "
         "into exactly ONE retrieval strategy.\n\n"
         "Available indexed documents:\n"
         f"{sources_list}\n\n"
         "Strategies:\n"
         "1. LLM_KNOWLEDGE — For greetings, casual chat, or general knowledge questions "
         "that have NOTHING to do with any documents.\n"
         "2. TARGETED_SEARCH:<source1> — When the query clearly relates to a specific document "
         "from the list above. Include the EXACT source name after the colon.\n"
         "3. ALL_DOCS — When the query asks about 'my documents', 'the resume', 'the uploaded file', "
         "or any broad research question where you are unsure which specific document is relevant.\n\n"
         "CRITICAL: If the user asks you to analyze, read, or summarize 'the document', 'the resume', "
         "or 'the file', you MUST output ALL_DOCS or TARGETED_SEARCH. NEVER use LLM_KNOWLEDGE for these.\n\n"
         "Reply with ONLY the strategy label, nothing else.\n\n"
         "Examples:\n"
         "- 'hey how are you' → LLM_KNOWLEDGE\n"
         "- 'what is machine learning' → LLM_KNOWLEDGE\n"
         "- 'summarize report.pdf' → TARGETED_SEARCH:report.pdf\n"
         "- 'what does main.py do' → TARGETED_SEARCH:main.py\n"
         "- 'compare findings across all papers' → ALL_DOCS\n"
         "- 'what do my documents say about AI' → ALL_DOCS"),
        ("human", "{query}")
    ])

    router_chain = router_prompt | llm
    t = time.time()
    response = router_chain.invoke({"query": query}).content.strip()
    print(f"[Router] Query: '{query}' → Decision: {response} ({time.time() - t:.1f}s)")

    response_clean = response.strip().upper()

    # Parse the router response
    if "TARGETED_SEARCH:" in response_clean:
        # Extract the part after TARGETED_SEARCH:
        idx = response.upper().find("TARGETED_SEARCH:")
        sources_str_original = response[idx + len("TARGETED_SEARCH:"):].strip()
        
        target_sources = [s.strip() for s in sources_str_original.split(",") if s.strip()]
        
        # Case-insensitive validation
        valid_sources = []
        indexed_lower = {s.lower(): s for s in indexed_sources}
        for ts in target_sources:
            if ts.lower() in indexed_lower:
                valid_sources.append(indexed_lower[ts.lower()])
                
        if valid_sources:
            return {"strategy": "targeted_search", "sources": valid_sources}
        else:
            print(f"[Router] Sources {target_sources} not found in index, falling back to ALL_DOCS")
            return {"strategy": "all_docs", "sources": []}
            
    elif "ALL_DOCS" in response_clean:
        return {"strategy": "all_docs", "sources": []}
        
    else:
        # Defaults to llm_knowledge for any other response
        return {"strategy": "llm_knowledge", "sources": []}
