import os
import uuid
import streamlit as st
from backend.utils import has_qdrant_data
from backend.logger import get_logger
from backend.rag import NODE_LABELS

logger = get_logger(__name__)


def render_chat_interface():
    """Render the main chat interface with live node-level streaming progress."""
    st.markdown('<h1 class="main-title">Chat with your Sources</h1>', unsafe_allow_html=True)

    if has_qdrant_data():
        st.markdown(
            '<p class="main-subtitle">🟢 Sources indexed — ask anything about your documents</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="main-subtitle">💡 Upload sources in the sidebar, or just chat — I\'ll use my own knowledge</p>',
            unsafe_allow_html=True,
        )

    # ── Per-session isolation ─────────────────────────────────────────────────
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info("New session: %s", st.session_state.session_id)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()

    # ── Render existing history ───────────────────────────────────────────────
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align: center; padding: 80px 20px; color: #64748b;">
                <div style="font-size: 3rem; margin-bottom: 16px;">🧠</div>
                <div style="font-size: 1.1rem; font-weight: 500; color: #94a3b8; margin-bottom: 8px;">
                    Agentic RAG Assistant
                </div>
                <div style="font-size: 0.85rem; max-width: 400px; margin: 0 auto; line-height: 1.6;">
                    Ask me anything — I'll intelligently route your query to the right source.<br>
                    <span style="color: #6366f1;">Documents</span> ·
                    <span style="color: #a78bfa;">LLM Knowledge</span> ·
                    <span style="color: #fbbf24;">Web Search</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    badge = msg.get("source_badge", "")
                    if badge:
                        badge_class = _badge_class(badge)
                        st.markdown(
                            f'<span class="source-badge {badge_class}">{badge}</span>',
                            unsafe_allow_html=True,
                        )
                    # Confidence badge for historical messages
                    if msg.get("retrieval_confidence") is not None:
                        _render_confidence_badge(
                            msg["retrieval_confidence"], msg.get("source_type", "")
                        )
                    # Latency breakdown for historical messages
                    if msg.get("timings"):
                        _render_latency(msg["timings"], msg.get("elapsed", 0))
                    # Follow-up chips for historical messages
                    if msg.get("followups"):
                        _render_followup_chips(msg["followups"], key_prefix=f"hist_{idx}")

    # ── Chat input + follow-up routing ───────────────────────────────────────
    # Render the chat input box (always visible)
    chat_input = st.chat_input("Ask a question across all your documents...")

    # A follow-up button click stores the question here; it takes priority
    prompt_to_run = st.session_state.pop("pending_followup", None) or chat_input

    if prompt_to_run:
        st.session_state.messages.append({"role": "user", "content": prompt_to_run})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt_to_run)

            with st.chat_message("assistant"):
                _handle_query(prompt_to_run)


# ── Badge helpers ─────────────────────────────────────────────────────────────

BADGE_MAP = {
    "llm_knowledge":   ("🧠 LLM Knowledge",   "badge-llm"),
    "targeted_search": ("🎯 Targeted Search",  "badge-target"),
    "local":           ("📂 All Documents",    "badge-docs"),
    "web":             ("🌐 Web Search",       "badge-web"),
    "hybrid_web":      ("🌐 Web + Documents",  "badge-web"),
    "cached":          ("⚡ Cached",           "badge-llm"),
}


def _badge_class(badge_text: str) -> str:
    if "LLM" in badge_text or "knowledge" in badge_text:
        return "badge-llm"
    if "Targeted" in badge_text or "targeted" in badge_text.lower():
        return "badge-target"
    if "Web" in badge_text or "web" in badge_text.lower():
        return "badge-web"
    return "badge-docs"


# ── Confidence badge renderer ─────────────────────────────────────────────────

def _render_confidence_badge(confidence: float, source_type: str):
    """Render a horizontal progress bar showing retrieval confidence (0–1).

    Only shown for document-based retrievals where a meaningful confidence
    score was produced by the cross-encoder or count heuristic.
    """
    if source_type not in {"local", "targeted_search"} or confidence <= 0:
        return

    if confidence >= 0.65:
        color, label = "#22c55e", "High"
    elif confidence >= 0.35:
        color, label = "#f59e0b", "Medium"
    else:
        color, label = "#ef4444", "Low"

    pct = int(confidence * 100)
    st.markdown(
        f'<div class="confidence-wrap">'
        f'  <span class="confidence-label">Confidence</span>'
        f'  <div class="confidence-track">'
        f'    <div class="confidence-fill" style="width:{pct}%;background:{color};"></div>'
        f'  </div>'
        f'  <span class="confidence-text" style="color:{color};">{label} {pct}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Latency breakdown renderer ────────────────────────────────────────────────

_NODE_DISPLAY_NAMES = {
    "cache_check":        "Cache",
    "guardrail":          "Guardrail",
    "router":             "Router",
    "retriever":          "Retriever",
    "evaluate_retrieval": "Eval",
    "rewrite_query":      "Rewrite",
    "web_search":         "Web Search",
    "generate_answer":    "LLM",
    "cache_store":        "Cache Store",
}


def _render_latency(timings: dict, total: float):
    """Render a compact per-node latency line below the answer."""
    if not timings:
        return
    parts = [
        f"{_NODE_DISPLAY_NAMES.get(node, node)} **{t}s**"
        for node, t in timings.items()
        if node in _NODE_DISPLAY_NAMES
    ]
    if parts:
        st.caption(f"⏱️ {total:.1f}s total  ·  " + "  ·  ".join(parts))


# ── Follow-up chips renderer ──────────────────────────────────────────────────

def _render_followup_chips(followups: list, key_prefix: str):
    """Render follow-up questions as pill-style chip buttons in a row.

    When a chip is clicked, the question is stored in pending_followup and
    a rerun is triggered so render_chat_interface picks it up as the next query.
    """
    if not followups:
        return

    st.markdown('<div class="followup-header">💡 You might also ask</div>', unsafe_allow_html=True)
    cols = st.columns(len(followups))
    for i, (col, question) in enumerate(zip(cols, followups)):
        with col:
            if st.button(question, key=f"{key_prefix}_fu_{i}", use_container_width=True):
                st.session_state.pending_followup = question
                st.rerun()


# ── Core query handler ────────────────────────────────────────────────────────

def _handle_query(prompt: str):
    """Run the agentic pipeline with live streaming progress, then render the answer."""
    from backend.rag import stream_agentic_response, generate_followup_questions

    chat_history = st.session_state.messages[:-1]
    session_id   = st.session_state.session_id

    answer    = ""
    meta      = {}
    had_error = False

    # Placeholder for the live-streamed answer (sits below the status box)
    answer_placeholder = st.empty()
    streaming_text     = ""

    # ── Live step-by-step progress ────────────────────────────────────────────
    with st.status("🔍 Thinking...", expanded=True) as status:
        try:
            for event_type, data in stream_agentic_response(prompt, chat_history, session_id):

                if event_type == "node":
                    label = NODE_LABELS.get(data, f"⚙️ {data}...")
                    status.update(label=label)

                elif event_type == "token":
                    # Append token and refresh the placeholder with a blinking cursor
                    streaming_text += data
                    answer_placeholder.markdown(streaming_text + " ▌")

                elif event_type == "answer":
                    answer = data
                    if answer:
                        # Final answer — remove the cursor and show clean text
                        answer_placeholder.markdown(answer)
                        status.update(label="✍️  Writing answer...")
                    else:
                        # Empty answer means INSUFFICIENT_CONTEXT → web retry incoming.
                        # Clear the placeholder so the next token stream starts fresh.
                        streaming_text = ""
                        answer_placeholder.empty()

                elif event_type == "meta":
                    meta = data

                elif event_type == "error":
                    st.error(data)
                    had_error = True
                    break

            if not had_error:
                elapsed = meta.get("elapsed", 0)
                status.update(
                    label=f"✅ Done in {elapsed:.1f}s",
                    state="complete",
                    expanded=False,
                )
        except Exception as exc:
            logger.exception("Chat stream error")
            st.error(f"An error occurred ({type(exc).__name__}). Please try again.")
            had_error = True

    if had_error or not answer:
        if not had_error:
            st.warning("Upload sources or ask a general question.")
        st.session_state.messages.append({"role": "assistant", "content": answer or "No response."})
        return

    # ── Render answer ─────────────────────────────────────────────────────────
    # The answer was already streamed live into answer_placeholder above.
    # Render badge, confidence, latency, and sources below it.

    source_type = meta.get("source_type", "")
    badge_text, badge_class = BADGE_MAP.get(source_type, ("📂 Documents", "badge-docs"))
    st.markdown(
        f'<span class="source-badge {badge_class}">{badge_text}</span>',
        unsafe_allow_html=True,
    )

    # ── Confidence badge ──────────────────────────────────────────────────────
    confidence = meta.get("retrieval_confidence", 0.0)
    _render_confidence_badge(confidence, source_type)

    # ── Latency breakdown ─────────────────────────────────────────────────────
    timings = meta.get("timings", {})
    elapsed = meta.get("elapsed", 0)
    _render_latency(timings, elapsed)

    # ── Sources expander ──────────────────────────────────────────────────────
    docs = meta.get("docs", [])
    if docs:
        with st.expander("📋 Sources & Metadata"):
            for i, doc in enumerate(docs):
                src    = doc.metadata.get("source", "Unknown")
                ftype  = doc.metadata.get("file_type", "unknown")
                origin = doc.metadata.get("origin", "local")
                icon   = "🌐" if origin == "web" else "📄"
                st.caption(f"**Source {i+1}** · {icon} `{ftype}` · {os.path.basename(str(src))}")
                st.text(doc.page_content[:300] + "...")
                if i < len(docs) - 1:
                    st.markdown("---")

    # ── Follow-up questions ───────────────────────────────────────────────────
    followups = generate_followup_questions(prompt, answer)
    msg_idx   = len(st.session_state.messages)   # index of the message we're about to append
    _render_followup_chips(followups, key_prefix=f"new_{msg_idx}")

    # ── Save to history ───────────────────────────────────────────────────────
    st.session_state.messages.append({
        "role":                 "assistant",
        "content":              answer,
        "source_badge":         badge_text,
        "source_type":          source_type,
        "context":              docs,
        "timings":              timings,
        "elapsed":              elapsed,
        "retrieval_confidence": confidence,
        "followups":            followups,
    })
