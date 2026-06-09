import os
import streamlit as st
from backend.utils import has_qdrant_data

def render_chat_interface():
    """Render the main chat interface and handle message processing."""
    st.markdown('<h1 class="main-title">Chat with your Sources</h1>', unsafe_allow_html=True)
    
    # Lightweight status bar — no heavy imports
    if has_qdrant_data():
        st.markdown('<p class="main-subtitle">🟢 Sources indexed — ask anything about your documents</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="main-subtitle">💡 Upload sources in the sidebar, or just chat — I\'ll use my own knowledge</p>', unsafe_allow_html=True)
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    chat_container = st.container()
    
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
                    <span style="color: #fbbf24;">Wikipedia</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                badge = msg.get("source_badge", "")
                if badge:
                    if "LLM" in badge or "knowledge" in badge:
                        badge_class = "badge-llm"
                    elif "targeted" in badge.lower():
                        badge_class = "badge-target"
                    elif "Wikipedia" in badge or "web" in badge.lower():
                        badge_class = "badge-web"
                    else:
                        badge_class = "badge-docs"
                    st.markdown(f'<span class="source-badge {badge_class}">{badge}</span>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question across all your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                from backend.rag import get_agentic_response
                
                with st.spinner("🔍 Thinking..."):
                    try:
                        chat_history = st.session_state.messages[:-1]
                        response = get_agentic_response(prompt, chat_history)
                        
                        if not response:
                            st.warning("Upload sources or ask a general question.")
                            st.session_state.messages.append({"role": "assistant", "content": "No documents indexed."})
                        else:
                            answer = response["answer"]
                            st.markdown(answer)
                            
                            source_type = response.get("source_type", "")
                            badge_map = {
                                "web": ("🌐 Wikipedia Fallback", "badge-web"),
                                "llm_knowledge": ("🧠 LLM Knowledge", "badge-llm"),
                                "targeted_search": ("🎯 Targeted Search", "badge-target"),
                                "local": ("📂 All Documents", "badge-docs"),
                            }
                            badge_text, badge_class = badge_map.get(source_type, ("📂 Documents", "badge-docs"))
                            st.markdown(f'<span class="source-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
                            
                            with st.expander("📋 Sources & Metadata"):
                                for i, doc in enumerate(response["context"]):
                                    src = doc.metadata.get("source", "Unknown")
                                    ftype = doc.metadata.get("file_type", "unknown")
                                    st.caption(f"**Source {i+1}** · `{ftype}` · {os.path.basename(str(src))}")
                                    st.text(doc.page_content[:300] + "...")
                                    if i < len(response["context"]) - 1:
                                        st.markdown("---")
                                    
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "source_badge": badge_text
                            })
                    except Exception as e:
                        st.error(f"Error: {e}")
