import streamlit as st
from backend.evaluation import evaluate_interaction

def render_dashboard():
    """Render the evaluation dashboard for quantifiable RAG metrics."""
    st.markdown('<h1 class="main-title">Evaluation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Quantifiable RAG Performance Metrics (RAGAS)</p>', unsafe_allow_html=True)
    
    if "messages" not in st.session_state or len(st.session_state.messages) < 2:
        st.info("No chat history available. Ask a question in the Chat tab first to evaluate the RAG pipeline.")
        return
        
    # Get the last user message and assistant message
    last_user = None
    last_assistant = None
    
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and not last_assistant:
            last_assistant = msg
        elif msg["role"] == "user" and not last_user and last_assistant:
            last_user = msg
            break
            
    if not last_user or not last_assistant:
        st.info("Incomplete interaction found.")
        return
        
    st.markdown("### Last Interaction")
    st.caption("**Question:** " + last_user["content"])
    st.caption("**Answer:** " + last_assistant["content"][:200] + "...")
    
    contexts = last_assistant.get("context", [])
    if not contexts:
        st.warning("No context was retrieved for this answer. Evaluation requires retrieved context.")
        return
        
    st.markdown("---")
    
    if st.button("📊 Evaluate Last Interaction", type="primary"):
        with st.spinner("Running RAGAS evaluation... This may take a minute depending on Groq API limits."):
            result = evaluate_interaction(
                question=last_user["content"],
                answer=last_assistant["content"],
                contexts=contexts
            )
            
            if result:
                st.success("✅ Evaluation complete!")
                
                # Display metrics in columns
                c1, c2, c3, c4 = st.columns(4)
                
                faithfulness = result.get('faithfulness', 0.0)
                relevancy = result.get('answer_relevancy', 0.0)
                precision = result.get('context_precision', 0.0)
                recall = result.get('context_recall', 0.0)
                
                c1.metric("Faithfulness", f"{faithfulness:.2f}", help="Did the answer come from retrieved docs?")
                c2.metric("Answer Relevancy", f"{relevancy:.2f}", help="Did the answer solve the question?")
                c3.metric("Context Precision", f"{precision:.2f}", help="How much retrieved context was useful?")
                c4.metric("Context Recall", f"{recall:.2f}", help="Did the retriever find all needed information?")
                
            else:
                st.error("Evaluation failed. Please check the console logs for detailed errors.")
