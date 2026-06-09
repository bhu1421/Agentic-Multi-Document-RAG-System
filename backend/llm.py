import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm():
    """Initialize the Groq LLM."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in .env")
        
    return ChatGroq(
        api_key=api_key,
        model_name="llama-3.1-8b-instant",  # Fast model for routing and generation
        temperature=0.1
    )
