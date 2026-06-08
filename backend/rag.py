# pyrefly: ignore [missing-import]
from langchain.chains import create_retrieval_chain
# pyrefly: ignore [missing-import]
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from backend.llm import get_llm
from backend.vectordb import get_vector_store
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

def get_agentic_response(query: str):
    store = get_vector_store()
    if not store:
        return None
        
    retriever = store.as_retriever(search_kwargs={"k": 10})
    llm = get_llm()
    
    # 1. Local RAG Phase
    system_prompt = (
        "You are an expert assistant. Use the following pieces of retrieved context to answer the question.\n"
        "If the context does not contain the answer, reply exactly with the word: WEB_SEARCH_REQUIRED\n"
        "Do not say anything else if the answer is not in the context.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    
    local_response = rag_chain.invoke({"input": query})
    answer = local_response["answer"].strip()
    
    # Check for our strict keyword OR common LLM refusal phrases
    refusal_phrases = ["unable to find", "don't know", "not mention", "not provided", "unable to locate", "WEB_SEARCH_REQUIRED"]
    needs_fallback = any(phrase.lower() in answer.lower() for phrase in refusal_phrases)
    
    # 2. Agentic Web Search Fallback Phase (Switched to Wikipedia for maximum stability)
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
            ("system", "You are an expert assistant. Answer the user's question using the following Wikipedia search results:\n\n{search_results}"),
            ("human", "{input}")
        ])
        
        web_chain = web_prompt | llm
        final_answer = web_chain.invoke({"search_results": search_results, "input": query})
        
        mock_doc = Document(
            page_content=search_results, 
            metadata={"source": "Wikipedia Agentic Search API", "file_type": "web_search"}
        )
        
        return {
            "answer": final_answer.content,
            "context": [mock_doc],
            "source_type": "web"
        }
    
    # If the answer was successfully found in local documents
    return {
        "answer": answer,
        "context": local_response["context"],
        "source_type": "local"
    }
