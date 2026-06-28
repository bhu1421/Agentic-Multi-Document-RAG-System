import os
import functools
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from backend import config

load_dotenv()


@functools.lru_cache(maxsize=4)
def get_llm(
    model_name: str = config.LLM_MODEL,
    temperature: float = config.LLM_TEMPERATURE,
) -> ChatGroq:
    """Return a cached Groq LLM instance.

    Caching is keyed on (model_name, temperature) so different configurations
    each get their own cached object, but repeated calls with the same args
    return the same instance — avoiding redundant object creation across the
    multiple graph nodes that call this function per query.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    return ChatGroq(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        request_timeout=config.LLM_REQUEST_TIMEOUT,
        max_retries=config.LLM_MAX_RETRIES,
    )
