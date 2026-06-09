import time
from langchain_core.prompts import ChatPromptTemplate

def rewrite_query(query: str, chat_history: list, llm) -> str:
    """
    Rewrite the user's query to be a standalone search query, 
    resolving any entities or pronouns based on the chat history.
    """
    if not chat_history:
        return query
        
    # Get the last 10 interactions (5 full turns) to retain deeper conversation context
    recent_history = chat_history[-10:]
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an intelligent query rewriter for a search engine. "
         "Given the following conversation history and a follow-up question, "
         "rewrite the follow-up question to be a standalone search query that includes all necessary context and entities from the history.\n"
         "If the question is already standalone and does not rely on history, return it exactly as is.\n"
         "Do not answer the question. Output ONLY the rewritten standalone question, without quotes or extra text."
         ),
        ("human", "Chat History:\n{history}\n\nFollow-up Question: {query}")
    ])
    
    chain = prompt | llm
    t = time.time()
    response = chain.invoke({"history": history_str, "query": query})
    standalone_query = response.content.strip()
    
    print(f"[Memory] Original: '{query}' -> Rewritten: '{standalone_query}' ({time.time() - t:.1f}s)")
    return standalone_query
