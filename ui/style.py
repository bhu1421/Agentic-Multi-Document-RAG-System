import streamlit as st

def apply_custom_styles():
    """Apply global custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        *, .stApp, .stMarkdown, .stTextInput input, .stTextArea textarea {
            font-family: 'Inter', sans-serif !important;
        }
        
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
            border-right: 1px solid rgba(99, 102, 241, 0.15);
        }
        
        .main-title {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0;
            letter-spacing: -0.5px;
        }
        .main-subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-top: 0;
            margin-bottom: 1.5rem;
        }
        
        .source-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            margin-top: 8px;
        }
        .badge-llm { background: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }
        .badge-target { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
        .badge-docs { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
        .badge-web { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
        
        .sidebar-section {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.08));
            border: 1px solid rgba(99, 102, 241, 0.15);
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 12px;
        }
        .sidebar-section h3 { margin: 0 0 4px 0; font-size: 0.9rem; color: #c7d2fe; }
        .sidebar-section p { margin: 0; font-size: 0.75rem; color: #64748b; }
        
        .stButton > button {
            border-radius: 10px;
            font-weight: 500;
            transition: all 0.2s ease;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }
        .stButton > button:hover {
            border-color: rgba(99, 102, 241, 0.6);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.15);
        }
        
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .status-active { background: #4ade80; box-shadow: 0 0 8px rgba(74, 222, 128, 0.5); }
        .status-empty { background: #64748b; }
        
        .custom-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
            border: none;
            margin: 16px 0;
        }
        
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
