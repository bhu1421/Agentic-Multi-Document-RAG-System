import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredMarkdownLoader

# Create uploaded_docs dir
DATA_DIR = Path("uploaded_docs")
DATA_DIR.mkdir(exist_ok=True)

def load_documents(file_path: str):
    """Load a document based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == ".pdf":
            loader = PyMuPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []
