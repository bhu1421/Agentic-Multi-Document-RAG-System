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

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                badge = msg.get("source_badge", "")
                if badge:
                    badge_class = _badge_class(badge)
                    st.markdown(
                        f'<span class="source-badge {badge_class}">{badge}</span>',
                        unsafe_allow_html=True,
                    )
                # Show stored latency breakdown for historical messages
                if msg.get("timings"):
                    _render_latency(msg["timings"], msg.get("elapsed", 0))

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question across all your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                _handle_query(prompt)


# ── Badge helpers ─────────────────────────────────────────────────────────────

BADGE_MAP = {
    "llm_knowledge":   ("🧠 LLM Knowledge",   "badge-llm"),
    "targeted_search": ("🎯 Targeted Search",  "badge-target"),
    "local":           ("📂 All Documents",    "badge-docs"),
    "web":             ("🌐 Web Search",       "badge-web"),
    "hybrid_web":      ("🌐 Web + Documents",  "badge-web"),
}


def _badge_class(badge_text: str) -> str:
    if "LLM" in badge_text or "knowledge" in badge_text:
        return "badge-llm"
    if "Targeted" in badge_text or "targeted" in badge_text.lower():
        return "badge-target"
    if "Web" in badge_text or "web" in badge_text.lower():
        return "badge-web"
    return "badge-docs"


# ── Latency breakdown renderer ────────────────────────────────────────────────

_NODE_DISPLAY_NAMES = {
    "router":          "Router",
    "retriever":       "Retriever",
    "web_search":      "Web Search",
    "generate_answer": "LLM",
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


# ── Core query handler ────────────────────────────────────────────────────────

def _handle_query(prompt: str):
    """Run the agentic pipeline with live streaming progress, then render the answer."""
    from backend.rag import stream_agentic_response

    chat_history = st.session_state.messages[:-1]
    session_id   = st.session_state.session_id

    answer    = ""
    meta      = {}
    had_error = False

    # ── Live step-by-step progress ────────────────────────────────────────────
    with st.status("🔍 Thinking...", expanded=True) as status:
        try:
            for event_type, data in stream_agentic_response(prompt, chat_history, session_id):

                if event_type == "node":
                    label = NODE_LABELS.get(data, f"⚙️ {data}...")
                    status.update(label=label)

                elif event_type == "answer":
                    answer = data
                    status.update(label="✍️  Writing answer...")

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
    st.markdown(answer)

    source_type = meta.get("source_type", "")
    badge_text, badge_class = BADGE_MAP.get(source_type, ("📂 Documents", "badge-docs"))
    st.markdown(
        f'<span class="source-badge {badge_class}">{badge_text}</span>',
        unsafe_allow_html=True,
    )

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

    st.session_state.messages.append({
        "role":         "assistant",
        "content":      answer,
        "source_badge": badge_text,
        "context":      docs,
        "timings":      timings,
        "elapsed":      elapsed,
    })
