import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    WebBaseLoader,
    GitLoader,
)
from backend import config
from backend.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(config.UPLOAD_DIR)
DATA_DIR.mkdir(exist_ok=True)

CLONED_REPOS_DIR = Path(config.CLONED_REPOS_DIR)
CLONED_REPOS_DIR.mkdir(exist_ok=True)


def _attach_user_metadata(docs, user_id: str | None = None):
    """Attach the current user to loaded documents so retrieval and cleanup stay scoped."""
    for doc in docs:
        if "source" in doc.metadata:
            doc.metadata["source"] = os.path.basename(doc.metadata["source"])
        doc.metadata["user_id"] = user_id or "public"
    return docs


def load_documents(file_path: str, user_id: str | None = None):
    """Load a document based on its file extension.

    Supported formats (actively exposed in the UI):
        .pdf  — PyMuPDF (fast, preserves page numbers)
        .txt  — Plain text
        .md   — Markdown (fed to the header-aware chunker)

    Supported formats (available in code, not in the UI file uploader):
        .docx / .doc  — Unstructured Word loader
        .pptx / .ppt  — Unstructured PowerPoint loader
        .xlsx         — Unstructured Excel loader
        .csv          — CSVLoader (one Document per row)
    """
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            loader = PyMuPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".md":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext in (".doc", ".docx"):
            loader = UnstructuredWordDocumentLoader(file_path, mode="elements", chunking_strategy="by_title")
        elif ext in (".ppt", ".pptx"):
            loader = UnstructuredPowerPointLoader(file_path, mode="elements", chunking_strategy="by_title")
        elif ext == ".xlsx":
            loader = UnstructuredExcelLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        else:
            logger.warning("[Loader] Skipping unsupported file extension: %s", ext)
            return []

        docs = loader.load()
        docs = _attach_user_metadata(docs, user_id)
        logger.info("[Loader] Loaded %d document(s) from '%s'", len(docs), os.path.basename(file_path))
        return docs

    except Exception as exc:
        logger.error("[Loader] Error loading '%s': %s", file_path, exc)
        return []


def load_web_urls(urls: list[str], user_id: str | None = None):
    """Load documents from a list of website URLs."""
    try:
        loader = WebBaseLoader(urls)
        docs = loader.load()
        docs = _attach_user_metadata(docs, user_id)
        logger.info("[Loader] Loaded %d page(s) from %d URL(s)", len(docs), len(urls))
        return docs
    except Exception as exc:
        logger.error("[Loader] Error loading web URLs: %s", exc)
        return []


def load_github_repo(repo_url: str, user_id: str | None = None):
    """Clone a GitHub repo and load its source/markdown files.

    Tries the default 'main' branch first, then falls back to 'master'
    so the function works with both old and new GitHub repositories.
    """
    if not repo_url:
        return []

    try:
        repo_name = repo_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        repo_root = CLONED_REPOS_DIR / (user_id or "public")
        repo_root.mkdir(parents=True, exist_ok=True)
        repo_path = str(repo_root / repo_name)

        for branch in ("main", "master", "develop"):
            try:
                loader = GitLoader(
                    clone_url=repo_url,
                    repo_path=repo_path,
                    branch=branch,
                    file_filter=lambda fp: fp.endswith((".py", ".md", ".txt", ".js", ".ts")),
                )
                docs = loader.load()
                docs = _attach_user_metadata(docs, user_id)
                logger.info(
                    "[Loader] Cloned '%s' (branch: %s) — %d file(s) loaded.",
                    repo_name, branch, len(docs),
                )
                return docs
            except Exception:
                logger.debug("[Loader] Branch '%s' not found for '%s', trying next.", branch, repo_name)
                continue

        logger.warning("[Loader] Could not load repo '%s' on any known branch.", repo_url)
        return []

    except Exception as exc:
        logger.error("[Loader] Error loading GitHub repo '%s': %s", repo_url, exc)
        return []
