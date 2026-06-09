import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    AsyncChromiumLoader,
    GitLoader
)

DATA_DIR = Path("uploaded_docs")
DATA_DIR.mkdir(exist_ok=True)

CLONED_REPOS_DIR = Path("cloned_repos")
CLONED_REPOS_DIR.mkdir(exist_ok=True)

def load_documents(file_path: str):
    """Load a document based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == ".pdf":
            loader = PyMuPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".md":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext in [".doc", ".docx"]:
            loader = UnstructuredWordDocumentLoader(file_path, mode="elements", chunking_strategy="by_title")
        elif ext in [".ppt", ".pptx"]:
            loader = UnstructuredPowerPointLoader(file_path, mode="elements", chunking_strategy="by_title")
        elif ext == ".xlsx":
            loader = UnstructuredExcelLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        else:
            print(f"Skipping unsupported file extension: {ext}")
            return []
            
        docs = loader.load()
        # Clean up the 'source' metadata to just the filename so LLM routing matches exactly
        for doc in docs:
            if "source" in doc.metadata:
                doc.metadata["source"] = os.path.basename(doc.metadata["source"])
        return docs
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

def load_web_urls(urls: list[str]):
    """Load documents from a list of website URLs."""
    try:
        import nest_asyncio
        nest_asyncio.apply()
        
        # Use AsyncChromiumLoader to render Javascript and return the raw HTML
        loader = AsyncChromiumLoader(urls)
        return loader.load()
    except Exception as e:
        print(f"Error loading web URLs: {e}")
        return []

def load_github_repo(repo_url: str):
    """Clone a GitHub repo and load its text/md files."""
    if not repo_url:
        return []
        
    try:
        repo_name = repo_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
            
        repo_path = str(CLONED_REPOS_DIR / repo_name)
        
        loader = GitLoader(
            clone_url=repo_url,
            repo_path=repo_path,
            branch="main",
            file_filter=lambda file_path: file_path.endswith((".py", ".md", ".txt"))
        )
        return loader.load()
    except Exception as e:
        print(f"Error loading GitHub Repo: {e}")
        return []
