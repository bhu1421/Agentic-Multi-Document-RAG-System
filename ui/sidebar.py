import os
import streamlit as st
from pathlib import Path
from backend.utils import (
    has_qdrant_data, 
    get_indexed_sources_ui, 
    perform_clear, 
    force_close_qdrant
)

DATA_DIR = Path("uploaded_docs")
CLONED_REPOS_DIR = Path("cloned_repos")

def render_sidebar():
    """Render the sidebar UI components."""
    with st.sidebar:
        st.markdown('<div class="sidebar-section"><h3>🧠 Agentic RAG</h3><p>Multi-source retrieval with intelligent routing</p></div>', unsafe_allow_html=True)
        
        # --- Data Ingestion ---
        st.markdown("#### 📥 Add Data Sources")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "docx", "pptx", "xlsx", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            st.caption(f"📎 {len(uploaded_files)} file(s) selected")
        
        web_urls_input = st.text_area(
            "🌐 Website URLs",
            placeholder="https://example.com/article\nhttps://docs.example.com",
            height=80
        )
        
        github_url_input = st.text_input(
            "🔗 GitHub Repository",
            placeholder="https://github.com/user/repo"
        )
        
        if st.button("⚡ Process All Sources", use_container_width=True, type="primary"):
            from backend.loader import load_documents, load_web_urls, load_github_repo
            from backend.chunker import chunk_documents
            from backend.vectordb import store_documents
            
            all_raw_docs = []
            
            if uploaded_files:
                with st.spinner("📄 Loading files..."):
                    for uploaded_file in uploaded_files:
                        file_path = DATA_DIR / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        docs = load_documents(str(file_path))
                        all_raw_docs.extend(docs)
                        
            if web_urls_input.strip():
                urls = [u.strip() for u in web_urls_input.split("\n") if u.strip()]
                with st.spinner(f"🌐 Scraping {len(urls)} URLs..."):
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
                        store_documents(chunks)
                    st.success(f"✅ {len(all_raw_docs)} sources → {len(chunks)} chunks")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # --- Database Status ---
        st.markdown("#### 🗄️ Database")
        
        if has_qdrant_data():
            st.markdown('<span class="status-dot status-active"></span> **Sources indexed**', unsafe_allow_html=True)
            if st.button("📋 View Sources"):
                indexed_sources = get_indexed_sources_ui()
                if indexed_sources:
                    for source in indexed_sources:
                        if str(source).startswith("http"):
                            display_name = source
                        else:
                            display_name = os.path.basename(source) if os.sep in source or "/" in source else source
                        st.caption(f"📄 {display_name}")
                else:
                    st.info("Collection is empty.")
        else:
            st.markdown('<span class="status-dot status-empty"></span> No sources indexed', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # --- Clear Database ---
        lock_file = os.path.join("local_qdrant", ".lock")
        has_lock = os.path.exists(lock_file)
        
        if st.button("🗑️ Clear Database", use_container_width=True):
            if has_lock:
                st.session_state["confirm_force_delete"] = True
            else:
                perform_clear()
        
        if st.session_state.get("confirm_force_delete"):
            st.warning("⚠️ Database is locked! Force-delete will close the Qdrant connection.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Force", use_container_width=True, type="primary"):
                    st.session_state.pop("confirm_force_delete", None)
                    force_close_qdrant()
                    perform_clear()
            with c2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.pop("confirm_force_delete", None)
                    st.rerun()
