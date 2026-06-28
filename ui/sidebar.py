import os
import re
import streamlit as st
from pathlib import Path
from backend import config
from backend.utils import has_qdrant_data, perform_clear, force_close_qdrant
from backend.vectordb import get_indexed_sources, delete_source
from backend.logger import get_logger

logger = get_logger(__name__)

DATA_DIR        = Path(config.UPLOAD_DIR)
CLONED_REPOS_DIR = Path(config.CLONED_REPOS_DIR)

MAX_FILE_SIZE_BYTES = config.MAX_FILE_SIZE_MB * 1024 * 1024


def _sanitize_filename(name: str) -> str:
    """Return a safe filename, stripping path components and dangerous characters.

    Prevents path-traversal attacks where an attacker names a file
    '../../etc/passwd' or similar.
    """
    safe = re.sub(r"[^\w.\- ]", "_", name)
    safe = safe.lstrip(".").strip()
    safe = Path(safe).name
    return safe or "upload"


def render_sidebar():
    """Render the sidebar UI components."""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-section"><h3>🧠 Agentic RAG</h3>'
            "<p>Multi-source retrieval with intelligent routing</p></div>",
            unsafe_allow_html=True,
        )

        # ── Data Ingestion ──────────────────────────────────────────────────
        st.markdown("#### 📥 Add Data Sources")

        uploaded_files = st.file_uploader(
            "Upload documents",
            # Actively supported: PDF, TXT, Markdown
            # DOCX/PPTX/XLSX/CSV are loaded in code but not advertised here
            # because they lack dedicated chunking strategies.
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            st.caption(f"📎 {len(uploaded_files)} file(s) selected")

        web_urls_input = st.text_area(
            "🌐 Website URLs",
            placeholder="https://example.com/article\nhttps://docs.example.com",
            height=80,
        )

        github_url_input = st.text_input(
            "🔗 GitHub Repository",
            placeholder="https://github.com/user/repo",
        )

        # ── Retrieval Settings ──────────────────────────────────────────────
        st.markdown("#### ⚙️ Retrieval Settings")

        use_reranker = st.toggle(
            "⚡ Cross-Encoder Reranking",
            value=False,
            help=(
                f"Enables {config.RERANKER_MODEL} to re-score retrieved chunks "
                f"for higher precision. Adds ~1–3s latency on CPU. "
                f"Recommended: ON when accuracy matters, OFF for faster responses."
            ),
        )
        os.environ["ENABLE_RERANKER"] = "true" if use_reranker else "false"

        if use_reranker:
            st.caption("🎯 Reranker ON — higher accuracy, slower response")
        else:
            st.caption("⚡ Reranker OFF — faster response")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Process Button ──────────────────────────────────────────────────
        if st.button("⚡ Process All Sources", use_container_width=True, type="primary"):
            from backend.loader import load_documents, load_web_urls, load_github_repo
            from backend.chunker import chunk_documents
            from backend.vectordb import get_vector_store

            all_raw_docs = []

            if uploaded_files:
                with st.spinner("📄 Loading files..."):
                    for uploaded_file in uploaded_files:

                        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                            st.error(
                                f"⚠️ **{uploaded_file.name}** is "
                                f"{uploaded_file.size / 1024 / 1024:.1f} MB — "
                                f"maximum allowed size is {config.MAX_FILE_SIZE_MB} MB. Skipped."
                            )
                            logger.warning(
                                "File rejected (too large): %s  %.1f MB",
                                uploaded_file.name,
                                uploaded_file.size / 1024 / 1024,
                            )
                            continue

                        safe_name = _sanitize_filename(uploaded_file.name)
                        if safe_name != uploaded_file.name:
                            logger.info(
                                "Filename sanitised: '%s' -> '%s'",
                                uploaded_file.name,
                                safe_name,
                            )

                        file_path = DATA_DIR / safe_name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        docs = load_documents(str(file_path))
                        all_raw_docs.extend(docs)

            if web_urls_input.strip():
                urls = [u.strip() for u in web_urls_input.split("\n") if u.strip()]
                with st.spinner(f"🌐 Scraping {len(urls)} URL(s)..."):
                    all_raw_docs.extend(load_web_urls(urls))

            if github_url_input.strip():
                with st.spinner("🔗 Cloning repo..."):
                    all_raw_docs.extend(load_github_repo(github_url_input.strip()))

            if not all_raw_docs:
                st.warning("No sources provided or parsed.")
            else:
                with st.spinner("✂️ Chunking..."):
                    chunks = chunk_documents(all_raw_docs)
                if not chunks:
                    st.warning("No readable text extracted.")
                else:
                    with st.spinner(f"🧮 Embedding {len(chunks)} chunks..."):
                        store = get_vector_store()
                        if store:
                            store.add_documents(chunks)
                            logger.info("[Qdrant] Stored %d chunks.", len(chunks))
                    st.success(f"✅ {len(all_raw_docs)} source(s) → {len(chunks)} chunks indexed")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Database Status ─────────────────────────────────────────────────
        st.markdown("#### 🗄️ Database")

        if has_qdrant_data():
            st.markdown('<span class="status-dot status-active"></span> **Sources indexed**', unsafe_allow_html=True)
            if st.button("📋 View Sources"):
                indexed_sources = get_indexed_sources()
                if indexed_sources:
                    for source in indexed_sources:
                        display_name = (
                            source if str(source).startswith("http")
                            else os.path.basename(source) if os.sep in source or "/" in source
                            else source
                        )
                        c1, c2 = st.columns([4, 1])
                        c1.caption(f"📄 {display_name}")
                        if c2.button("❌", key=f"del_{source}", help="Delete this source"):
                            delete_source(source)
                            st.rerun()
                else:
                    st.info("Collection is empty.")
        else:
            st.markdown('<span class="status-dot status-empty"></span> No sources indexed', unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Observability ───────────────────────────────────────────────────
        st.markdown("#### 🔭 Observability")
        env_api_key = os.getenv("LANGCHAIN_API_KEY", "")

        tracing_enabled = st.toggle("Enable LangSmith Tracing", value=False)

        if env_api_key and tracing_enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            if "LANGCHAIN_PROJECT" not in os.environ:
                os.environ["LANGCHAIN_PROJECT"] = "Agentic_RAG_System"
            st.caption("✅ Tracing Active — view at smith.langchain.com")
        elif not env_api_key and tracing_enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            st.caption("⚠️ Set LANGCHAIN_API_KEY in .env to enable tracing.")
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            st.caption("⏸️ Tracing paused.")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Clear Database ──────────────────────────────────────────────────
        if st.button("🗑️ Clear Database", use_container_width=True):
            perform_clear()
