"""
Redis Cache — optional response caching for the Agentic RAG pipeline.

Two LangGraph nodes:
    cache_check_node()   — checks Redis for a cached answer (runs first)
    cache_store_node()   — stores the answer in Redis (runs last)

Design: fully optional. If redis is not installed or the server is
unreachable, the system continues working normally — no crashes.
"""

import hashlib
import json
import time
import os
from datetime import datetime
from backend import config
from backend.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Redis Client (lazy, fault-tolerant)
# ──────────────────────────────────────────────

_redis_client = None
_redis_checked = False


def get_redis_client():
    """Return a Redis client, or None if Redis is unavailable.

    Lazy-initialised and cached.  Never raises — returns None on any failure.
    """
    global _redis_client, _redis_checked

    if _redis_checked:
        return _redis_client

    _redis_checked = True

    if not config.CACHE_ENABLED:
        logger.info("[Cache] Caching is disabled via config")
        return None

    try:
        import redis
        client = redis.Redis.from_url(
            config.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        logger.info("[Cache] Redis connected: %s", config.REDIS_URL)
        _redis_client = client
        return _redis_client
    except ImportError:
        logger.warning("[Cache] redis package not installed — caching disabled")
    except Exception as exc:
        logger.warning("[Cache] Redis unavailable (%s) — caching disabled", exc)

    return None


def reset_redis_client():
    """Reset the cached client — useful if Redis becomes available later."""
    global _redis_client, _redis_checked
    _redis_client = None
    _redis_checked = False


# ──────────────────────────────────────────────
# Cache Key
# ──────────────────────────────────────────────

def build_cache_key(user_id: str, query: str) -> str:
    """Build a deterministic cache key from user_id and normalised query.

    Normalisation: strip, lowercase, collapse whitespace.
    Key format: rag:cache:<sha256>
    """
    normalised = " ".join(query.strip().lower().split())
    raw = f"{user_id}:{normalised}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"rag:cache:{digest}"


# ──────────────────────────────────────────────
# Node — Cache Check (runs at START)
# ──────────────────────────────────────────────

def cache_check_node(state: dict) -> dict:
    """LangGraph node: check Redis for a cached response.

    Returns cache_hit=True and the cached answer if found,
    otherwise cache_hit=False to let the pipeline continue.
    """
    t = time.time()
    client = get_redis_client()

    if client is None:
        return {
            "cache_hit": False,
            "timings": {**state.get("timings", {}), "cache_check": round(time.time() - t, 2)},
        }

    user_id = state.get("user_id", "public")
    query = state.get("query", "")
    key = build_cache_key(user_id, query)

    try:
        cached = client.get(key)
        if cached:
            data = json.loads(cached)
            elapsed = round(time.time() - t, 2)
            logger.info(
                "[Cache] HIT — key=%s (cached at %s)",
                key[:30], data.get("timestamp", "?"),
            )
            return {
                "cache_hit": True,
                "answer": data["answer"],
                "source_type": data.get("source_type", "cached"),
                "timings": {**state.get("timings", {}), "cache_check": elapsed},
            }
    except Exception as exc:
        logger.warning("[Cache] Read failed: %s", exc)

    elapsed = round(time.time() - t, 2)
    logger.info("[Cache] MISS — key=%s (%.2fs)", key[:30], elapsed)
    return {
        "cache_hit": False,
        "timings": {**state.get("timings", {}), "cache_check": elapsed},
    }


# ──────────────────────────────────────────────
# Node — Cache Store (runs at END)
# ──────────────────────────────────────────────

def cache_store_node(state: dict) -> dict:
    """LangGraph node: store the generated answer in Redis with TTL.

    Skips caching if:
    - Redis is unavailable
    - The answer was itself a cache hit
    - The answer is empty or a block message
    """
    t = time.time()
    client = get_redis_client()

    if client is None:
        return {
            "timings": {**state.get("timings", {}), "cache_store": round(time.time() - t, 2)},
        }

    # Don't re-cache a cache hit
    if state.get("cache_hit", False):
        return {
            "timings": {**state.get("timings", {}), "cache_store": round(time.time() - t, 2)},
        }

    # Don't cache blocked responses
    if state.get("guardrail_result") == "blocked":
        return {
            "timings": {**state.get("timings", {}), "cache_store": round(time.time() - t, 2)},
        }

    answer = state.get("answer", "")
    if not answer or answer == "I couldn't generate an answer.":
        return {
            "timings": {**state.get("timings", {}), "cache_store": round(time.time() - t, 2)},
        }

    user_id = state.get("user_id", "public")
    query = state.get("original_query") or state.get("query", "")
    key = build_cache_key(user_id, query)

    payload = json.dumps({
        "answer": answer,
        "source_type": state.get("source_type", ""),
        "timestamp": datetime.now().isoformat(),
        "latency": sum(state.get("timings", {}).values()),
    })

    try:
        client.setex(key, config.CACHE_TTL_SECONDS, payload)
        logger.info("[Cache] STORED — key=%s ttl=%ds", key[:30], config.CACHE_TTL_SECONDS)
    except Exception as exc:
        logger.warning("[Cache] Write failed: %s", exc)

    return {
        "timings": {**state.get("timings", {}), "cache_store": round(time.time() - t, 2)},
    }


# ──────────────────────────────────────────────
# Conditional Edge — Route After Cache Check
# ──────────────────────────────────────────────

def route_after_cache(state: dict) -> str:
    """If cache hit, go straight to END. Otherwise, continue to guardrail."""
    if state.get("cache_hit", False):
        return "__end__"
    return "guardrail"
