"""
Guardrail Node — validates every incoming query before routing.

Architecture:
    GuardrailChecker    — stateless class with individual check methods
    guardrail_node()    — LangGraph node that orchestrates the checks

Each check method returns (is_blocked: bool, reason: str).
Adding a new check = adding one method + one entry in CHECKS list.
"""

import re
import time
from backend import config
from backend.logger import get_logger

logger = get_logger(__name__)


# ── Individual Check Functions ────────────────────────────────────────────────
# Each returns (is_blocked: bool, reason: str).
# Keeping them as module-level functions makes them easy to test and extend.


def check_empty(query: str, **_kwargs) -> tuple[bool, str]:
    """Reject empty or whitespace-only queries."""
    if not query or not query.strip():
        return True, "Empty query"
    return False, ""


# ── Conversational / Greeting Patterns ───────────────────────────────────────
# Short greetings and casual messages should always pass through so the LLM
# can respond naturally — no LLM off-topic check needed for these.

_GREETING_WORDS = re.compile(
    r"^\s*(hi|hello|hey|howdy|greetings|sup|yo|hiya)[!.,?\s]",
    re.IGNORECASE,
)

_GREETING_PATTERNS = re.compile(
    r"^\s*("
    r"hi|hello|hey|howdy|greetings|sup|yo|hiya|"           # greetings
    r"good\s+(morning|afternoon|evening|night)|"            # time greetings
    r"how\s+are\s+you|how\s+r\s+u|how\s+do\s+you\s+do|"  # pleasantries
    r"(i\s+(am|'m)|my\s+name\s+is|call\s+me|i\s+go\s+by)\s+\w+|"  # intros
    r"what('s|\s+is)\s+your\s+name|who\s+are\s+you|"      # identity questions
    r"thanks?|thank\s+you|thx|ty|cheers|"                  # gratitude
    r"ok|okay|sure|alright|cool|great|nice|wow|"           # acknowledgements
    r"bye|goodbye|see\s+you|later|cya"                      # farewells
    r")[!.,?\s]*$",
    re.IGNORECASE,
)


def check_conversational(query: str, **_kwargs) -> tuple[bool, str]:
    """Allow short conversational/greeting messages to pass without LLM check.

    Queries that match greeting/casual patterns are guaranteed NOT to be
    blocked as off-topic — the LLM will respond naturally to them.
    This check always returns (False, '') meaning 'not blocked'.
    It is added to the CHECKS list BEFORE check_off_topic so that matched
    queries set a pass-flag that prevents the LLM check from running.
    """
    stripped = query.strip()

    # Very short queries (≤20 chars) → fast pass (no LLM check needed)
    if len(stripped) <= 20:
        logger.info("[Guardrail] Short query (%d chars) — fast pass", len(stripped))
        return False, ""

    # Queries that start with a greeting word → fast pass
    # Catches: "hey i am bhuvan", "hi, what do you know about...", etc.
    if _GREETING_WORDS.match(stripped):
        logger.info("[Guardrail] Greeting-prefixed query — fast pass")
        return False, ""

    # Personal intro patterns → fast pass
    if _GREETING_PATTERNS.match(stripped):
        logger.info("[Guardrail] Greeting-style intro query — fast pass")
        return False, ""

    return False, ""



def check_length(query: str, **_kwargs) -> tuple[bool, str]:
    """Reject queries exceeding the configured character limit."""
    max_len = config.GUARDRAIL_MAX_QUERY_LENGTH
    if len(query) > max_len:
        return True, f"Query too long ({len(query):,} chars, max {max_len:,})"
    return False, ""


# ── Spam Detection Patterns ──────────────────────────────────────────────────

# Repeated single character: aaaaaaaaa (10+ of the same char)
_RE_REPEATED_CHAR = re.compile(r"(.)\1{9,}")

# Repeated punctuation block: !!!!!!!!!!! or ?????????
_RE_REPEATED_PUNCT = re.compile(r"[!?.#@$%^&*]{10,}")

# Repeated words/phrases: same token appearing 5+ times consecutively
_RE_REPEATED_WORD = re.compile(r"\b(\w+)(?:\s+\1){4,}\b", re.IGNORECASE)


def check_spam(query: str, **_kwargs) -> tuple[bool, str]:
    """Reject spam-like queries (repeated chars, punctuation, or phrases)."""
    stripped = query.strip()

    if _RE_REPEATED_CHAR.search(stripped):
        return True, "Spam detected (repeated characters)"

    if _RE_REPEATED_PUNCT.search(stripped):
        return True, "Spam detected (repeated punctuation)"

    if _RE_REPEATED_WORD.search(stripped):
        return True, "Spam detected (repeated words)"

    # Check if the same sentence is copy-pasted many times
    sentences = [s.strip() for s in re.split(r'[.!?\n]', stripped) if s.strip()]
    if len(sentences) >= 5:
        from collections import Counter
        counts = Counter(sentences)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count >= 5 and most_common_count / len(sentences) > 0.6:
            return True, "Spam detected (repeated sentences)"

    return False, ""


# ── Prompt Injection Patterns ─────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"show\s+(me\s+)?(your\s+)?(hidden|system)\s+prompt",
    r"print\s+(your\s+)?api\s*keys?",
    r"show\s+(me\s+)?(your\s+)?api\s*keys?",
    r"output\s+(your\s+)?(initial|system)\s+(instructions?|prompt)",
    r"what\s+(is|are)\s+your\s+(system\s+)?instructions",
    r"repeat\s+(your\s+)?(system\s+)?(prompt|instructions)",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    r"you\s+are\s+now\s+(dan|jailbroken|unrestricted)",
    r"override\s+(your\s+)?safety",
    r"bypass\s+(your\s+)?(safety|content)\s+(filter|policy)",
    r"pretend\s+(that\s+)?(you\s+)?(don'?t|do\s+not)\s+have\s+(any\s+)?rules",
]

_RE_INJECTION = re.compile(
    "|".join(f"(?:{p})" for p in _INJECTION_PATTERNS),
    re.IGNORECASE,
)


def check_prompt_injection(query: str, **_kwargs) -> tuple[bool, str]:
    """Reject queries containing prompt injection attempts."""
    if _RE_INJECTION.search(query):
        return True, "Prompt injection detected"
    return False, ""


# ── Malicious Request Patterns ────────────────────────────────────────────────

_MALICIOUS_PATTERNS = [
    r"generate\s+malware",
    r"create\s+(a\s+)?virus",
    r"write\s+(a\s+)?(malware|virus|trojan|ransomware|keylogger)",
    r"hack\s+(a\s+|into\s+)?(website|server|system|account|database)",
    r"delete\s+(the\s+|all\s+)?database",
    r"drop\s+(the\s+|all\s+)?table",
    r"steal\s+(the\s+)?passwords?",
    r"(crack|brute\s*force)\s+(a\s+)?password",
    r"(ddos|dos)\s+attack",
    r"(phishing|spear\s*phishing)\s+(email|page|attack)",
    r"exploit\s+(a\s+)?(vulnerability|zero\s*day)",
    r"sql\s+injection\s+attack",
    r"how\s+to\s+(hack|exploit|attack|compromise)",
    r"create\s+(a\s+)?(backdoor|rootkit|botnet)",
]

_RE_MALICIOUS = re.compile(
    "|".join(f"(?:{p})" for p in _MALICIOUS_PATTERNS),
    re.IGNORECASE,
)


def check_malicious(query: str, **_kwargs) -> tuple[bool, str]:
    """Reject queries requesting harmful or illegal activities."""
    if _RE_MALICIOUS.search(query):
        return True, "Malicious request detected"
    return False, ""


def check_off_topic(query: str, llm=None, **_kwargs) -> tuple[bool, str]:
    """Use the LLM to detect completely off-topic task requests.

    Only runs when config.GUARDRAIL_ENABLE_LLM_CHECK is True and an LLM
    instance is provided.  Returns (True, reason) if the query is a task
    request unrelated to document analysis, knowledge retrieval, OR
    general conversation.
    """
    if not config.GUARDRAIL_ENABLE_LLM_CHECK or llm is None:
        return False, ""

    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field
    from typing import Literal

    class TopicCheck(BaseModel):
        """Structured output for off-topic detection."""
        verdict: Literal["relevant", "off_topic"] = Field(
            description=(
                "relevant — the query is a general knowledge question, a document/research "
                "question, a greeting, a casual conversation message, a personal introduction, "
                "or anything a helpful AI assistant would normally respond to. "
                "off_topic — ONLY flag this if the query explicitly demands a harmful, "
                "purely creative-fiction task (e.g. write me a poem about dragons), or "
                "a very specific unrelated action (e.g. write a recipe for cake). "
                "When in doubt, choose relevant."
            )
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a lenient query classifier for an AI assistant. "
         "Your job is to decide if a query should be BLOCKED.\n\n"
         "Mark as RELEVANT (do NOT block):\n"
         "- Greetings and casual conversation ('hi', 'hello', 'how are you')\n"
         "- Personal introductions ('I am Bhuvan', 'my name is ...')\n"
         "- General knowledge questions ('what is ML?', 'explain Python')\n"
         "- Document/research questions\n"
         "- Anything a normal AI assistant would helpfully answer\n\n"
         "Mark as OFF_TOPIC (block) ONLY IF:\n"
         "- The user asks for something clearly harmful\n"
         "- The user demands an unrelated creative task the assistant obviously cannot help with\n\n"
         "DEFAULT TO RELEVANT. Only block when you are very sure."),
        ("human", "{query}"),
    ])

    try:
        structured_llm = llm.with_structured_output(TopicCheck)
        result = (prompt | structured_llm).invoke({"query": query})
        if isinstance(result, dict):
            result = TopicCheck(**result)
        if result.verdict == "off_topic":
            return True, "Off-topic request"
    except Exception as exc:
        # If the LLM check fails, let the query through (fail open)
        logger.warning("[Guardrail] Off-topic check failed: %s", exc)

    return False, ""


# ── Ordered list of all checks ────────────────────────────────────────────────
# Cheap checks first, LLM-based check last.
# To add a new check: define a function above, then append it here.

CHECKS = [
    check_empty,
    check_length,
    check_spam,
    check_prompt_injection,
    check_malicious,
    check_conversational,   # Fast-pass greetings/casual — runs before LLM check
    check_off_topic,        # LLM-based — runs last (skipped for conversational queries)
]


# ── Friendly block messages ───────────────────────────────────────────────────

_BLOCK_MESSAGES = {
    "Empty query":
        "It looks like you sent an empty message. Please type a question and try again! 😊",
    "Prompt injection detected":
        "I detected a prompt injection attempt. I can't process this request. "
        "Please ask a genuine question about your documents or general knowledge.",
    "Malicious request detected":
        "I'm unable to help with that kind of request. "
        "I'm designed to answer questions about your documents and general knowledge.",
    "Off-topic request":
        "That request seems unrelated to document analysis or knowledge retrieval. "
        "I'm best at answering questions about your uploaded documents or general knowledge topics. "
        "Try asking something like *'Summarize my document'* or *'What is machine learning?'*",
}


def _get_block_message(reason: str) -> str:
    """Return a user-friendly message for a given block reason."""
    for key, msg in _BLOCK_MESSAGES.items():
        if key.lower() in reason.lower():
            return msg
    # Fallback for spam, length, and any new checks
    return (
        f"I couldn't process your request ({reason}). "
        "Please rephrase and try again."
    )


# ── LangGraph Node ────────────────────────────────────────────────────────────

def guardrail_node(state: dict) -> dict:
    """LangGraph node: validate the incoming query through all guardrail checks.

    Returns:
        guardrail_result: "pass" | "blocked"
        guardrail_reason: reason string (empty on pass)
        answer: friendly rejection message (empty on pass)
    """
    t = time.time()
    query = state.get("query", "")

    # For the off-topic check, we need an LLM instance
    llm = None
    if config.GUARDRAIL_ENABLE_LLM_CHECK:
        try:
            from backend.llm import get_llm
            llm = get_llm()
        except Exception:
            pass  # If LLM is unavailable, skip LLM-based checks

    for check_fn in CHECKS:
        is_blocked, reason = check_fn(query, llm=llm)
        if is_blocked:
            elapsed = round(time.time() - t, 2)
            logger.warning(
                "[Guardrail] BLOCKED — reason: '%s' | query: '%.80s...'",
                reason, query,
            )
            return {
                "guardrail_result": "blocked",
                "guardrail_reason": reason,
                "answer": _get_block_message(reason),
                "timings": {**state.get("timings", {}), "guardrail": elapsed},
            }

    elapsed = round(time.time() - t, 2)
    logger.info("[Guardrail] PASS (%.2fs)", elapsed)
    return {
        "guardrail_result": "pass",
        "guardrail_reason": "",
        "timings": {**state.get("timings", {}), "guardrail": elapsed},
    }
