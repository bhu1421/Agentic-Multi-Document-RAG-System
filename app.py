import os
import streamlit as st

# Workarounds for Streamlit/Torch watcher issues
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import torch
if hasattr(torch, "classes") and hasattr(torch.classes, "__path__"):
    torch.classes.__path__ = []

from backend.loader import load_documents, DATA_DIR
from backend.chunker import chunk_documents
from backend.vectordb import store_documents, get_vector_store
from backend.rag import get_rag_chain

st.set_page_config(page_title="Phase 1: Foundation RAG", page_icon="📄")

st.title("📄 Foundation RAG System")
st.markdown("Upload a PDF, TXT, or Markdown file to chat with it.")

# File Uploader
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])

if uploaded_file is not None:
    # Save file
    file_path = DATA_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    if st.button("Process Document"):
        with st.spinner("Extracting text..."):
            docs = load_documents(str(file_path))
            
        if not docs:
            st.error("Failed to load document.")
        else:
            with st.spinner("Chunking text..."):
                chunks = chunk_documents(docs)
                
            with st.spinner(f"Embedding {len(chunks)} chunks into Qdrant..."):
                store_documents(chunks)
                
            st.success("Document processed and stored successfully! You can now chat.")

st.divider()

if st.button("🗑️ Clear Entire Vector Database & Uploaded Docs"):
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
            os.makedirs("uploaded_docs") # Recreate empty folder for future uploads
            cleared_anything = True
        except Exception as e:
            st.error(f"Failed to clear uploaded docs: {e}")
            
    if cleared_anything:
        st.success("Database and uploaded documents wiped clean! The AI has forgotten all previous files.")
    else:
        st.info("Everything is already empty.")

st.divider()

# Chat Interface
st.subheader("Chat with Document")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the uploaded document..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        rag_chain = get_rag_chain()
        if not rag_chain:
            st.warning("Please upload and process a document first.")
            st.session_state.messages.append({"role": "assistant", "content": "No documents indexed."})
        else:
            with st.spinner("Retrieving answers..."):
                try:
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # Show sources
                    with st.expander("Sources"):
                        for i, doc in enumerate(response["context"]):
                            st.caption(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}")
                            st.text(doc.page_content[:200] + "...")
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error answering question: {e}")
