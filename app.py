import os
import streamlit as st

# Workarounds for Streamlit/Torch watcher issues
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import torch
if hasattr(torch, "classes") and hasattr(torch.classes, "__path__"):
    torch.classes.__path__ = []

from backend.loader import load_documents, load_web_urls, load_github_repo, DATA_DIR, CLONED_REPOS_DIR
from backend.chunker import chunk_documents
from backend.vectordb import store_documents, get_vector_store, get_indexed_sources, delete_source
from backend.rag import get_agentic_response

st.set_page_config(page_title="Phase 2: Multi-Source RAG", page_icon="🗂️")

st.title("🗂️ Multi-Source RAG System")
st.markdown("Upload multiple documents, website URLs, and GitHub repos simultaneously!")

# --- Data Ingestion Section ---
with st.expander("📥 Add Data Sources", expanded=True):
    
    # 1. File Uploader (Multi-file)
    uploaded_files = st.file_uploader(
        "Upload documents", 
        type=["pdf", "txt", "md", "docx", "pptx", "xlsx", "csv"],
        accept_multiple_files=True
    )
    
    # 2. Web URLs
    web_urls_input = st.text_area("Website URLs (one per line)", placeholder="https://example.com/article1\nhttps://example.com/article2")
    
    # 3. GitHub Repo
    github_url_input = st.text_input("GitHub Repository URL", placeholder="https://github.com/username/repo")
    
    if st.button("Process All Sources"):
        all_raw_docs = []
        
        # Process Files
        if uploaded_files:
            with st.spinner(f"Loading {len(uploaded_files)} local files..."):
                for uploaded_file in uploaded_files:
                    file_path = DATA_DIR / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    docs = load_documents(str(file_path))
                    all_raw_docs.extend(docs)
                    
        # Process URLs
        if web_urls_input.strip():
            urls = [url.strip() for url in web_urls_input.split("\n") if url.strip()]
            with st.spinner(f"Scraping {len(urls)} websites..."):
                web_docs = load_web_urls(urls)
                all_raw_docs.extend(web_docs)
                
        # Process GitHub
        if github_url_input.strip():
            with st.spinner(f"Cloning & parsing GitHub Repo..."):
                git_docs = load_github_repo(github_url_input.strip())
                all_raw_docs.extend(git_docs)
                
        if not all_raw_docs:
            st.warning("No data sources were provided or parsed successfully.")
        else:
            with st.spinner("Standardizing and chunking text..."):
                chunks = chunk_documents(all_raw_docs)
                
            if not chunks:
                st.warning("Could not extract any readable text from these sources! (They might be empty, password-protected, or a blocked website).")
            else:
                with st.spinner(f"Embedding {len(chunks)} chunks into Qdrant..."):
                    store_documents(chunks)
                
                st.success(f"Successfully processed {len(all_raw_docs)} files/pages into {len(chunks)} chunks! You can now chat.")

st.divider()

# --- Database Management Section ---
with st.expander("🗄️ Database Management", expanded=False):
    st.markdown("Selectively delete files or URLs from your vector database.")
    
    indexed_sources = get_indexed_sources()
    
    if not indexed_sources:
        st.info("The database is currently empty.")
    else:
        for source in indexed_sources:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.write(f"📄 {source}")
            with col2:
                if st.button("🗑️ Delete", key=f"del_{source}"):
                    success = delete_source(source)
                    if success:
                        st.success(f"Deleted {source}!")
                        st.rerun()
                    else:
                        st.error("Failed to delete.")
                        
    st.divider()
    if st.button("🚨 Clear Entire Vector Database & Uploaded Docs"):
        import shutil
        cleared_anything = False
        
        if os.path.exists("local_qdrant"):
            try:
                shutil.rmtree("local_qdrant")
                cleared_anything = True
            except Exception as e:
                st.error(f"Failed to clear database: {e}")
                
        if os.path.exists("uploaded_docs"):
            try:
                shutil.rmtree("uploaded_docs")
                os.makedirs("uploaded_docs") 
                cleared_anything = True
            except Exception as e:
                st.error(f"Failed to clear uploaded docs: {e}")
                
        if os.path.exists("cloned_repos"):
            try:
                shutil.rmtree("cloned_repos")
                os.makedirs("cloned_repos")
                cleared_anything = True
            except Exception as e:
                st.error(f"Failed to clear cloned repos: {e}")
                
        if cleared_anything:
            st.session_state.messages = []
            st.success("Entire Database and all documents successfully wiped!")
            st.rerun()
        else:
            st.info("Everything is already empty.")

st.divider()

# --- Chat Interface ---
st.subheader("Chat with your Sources")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question across all your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        from backend.rag import get_agentic_response
        
        with st.spinner("Analyzing documents & querying agent..."):
            try:
                response = get_agentic_response(prompt)
                
                if not response:
                    st.warning("Please upload and process a data source first.")
                    st.session_state.messages.append({"role": "assistant", "content": "No documents indexed."})
                else:
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    if response.get("source_type") == "web":
                        st.info("🌐 Answer generated via Wikipedia Agentic API Fallback.")
                    else:
                        st.success("📂 Answer retrieved securely from local documents.")
                    
                    # Show sources
                    with st.expander("Sources & Metadata"):
                        for i, doc in enumerate(response["context"]):
                            st.caption(f"Source {i+1} [{doc.metadata.get('file_type', 'unknown')}]: {doc.metadata.get('source', 'Unknown')}")
                            st.text(doc.page_content[:500] + "...")
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error answering question: {e}")
