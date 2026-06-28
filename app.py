import os
import streamlit as st
from ui.style import apply_custom_styles
from ui.sidebar import render_sidebar
from ui.chat import render_chat_interface

# Set env var early to disable file watcher for performance
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# Page configuration
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render UI components
apply_custom_styles()
render_sidebar()
render_chat_interface()
