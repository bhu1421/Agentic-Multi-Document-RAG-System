import uuid
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    Language,
)
from backend import config


def chunk_documents(documents):
    """Split documents and enforce the Metadata Standard using context-aware chunking.

    Strategy selection is driven by file_type:
    - PDF/TXT/unknown  → RecursiveCharacterTextSplitter (generic fallback)
    - Markdown         → Header-aware split, then recursive (preserves document structure)
    - HTML/Web         → HTML header split, then recursive
    - CSV              → No-op (already one row per Document from CSVLoader)
    - Source code      → Language-aware splitter (respects function/class boundaries)
    """
    standardized_chunks = []

    # ── Splitters ─────────────────────────────────────────────────────────────
    base_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",   "Header 1"),
            ("##",  "Header 2"),
            ("###", "Header 3"),
        ]
    )

    html_splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=[
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
        ]
    )

    language_map = {
        "py": Language.PYTHON, "js": Language.JS,  "ts": Language.TS,
        "go": Language.GO,     "java": Language.JAVA, "cpp": Language.CPP,
        "c":  Language.C,      "cs": Language.CSHARP, "rb": Language.RUBY,
        "rs": Language.RUST,   "php": Language.PHP,
    }

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", None)

        if source.startswith("http"):
            file_type = "web"
        else:
            file_type = source.split(".")[-1].lower() if "." in source else "unknown"

        # Every original document gets a unique parent_id.
        # All chunks produced from it share this ID, enabling hierarchical retrieval
        # to later fetch all siblings of any matched chunk.
        parent_id = str(uuid.uuid4())
        chunks = []

        if file_type == "pdf":
            chunks = base_splitter.split_documents([doc])
        elif file_type == "md":
            md_docs = md_splitter.split_text(doc.page_content)
            chunks = base_splitter.split_documents(md_docs)
            for c in chunks:
                c.metadata.update(doc.metadata)
        elif file_type in ("html", "web"):
            try:
                html_docs = html_splitter.split_text(doc.page_content)
                chunks = base_splitter.split_documents(html_docs)
                for c in chunks:
                    c.metadata.update(doc.metadata)
            except Exception:
                chunks = base_splitter.split_documents([doc])
        elif file_type == "csv":
            chunks = [doc]  # CSVLoader already yields one Document per row
        elif file_type in language_map:
            code_splitter = RecursiveCharacterTextSplitter.from_language(
                language=language_map[file_type],
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
            )
            chunks = code_splitter.split_documents([doc])
        else:
            chunks = base_splitter.split_documents([doc])

        # ── Standardise metadata on every chunk ───────────────────────────────
        for chunk in chunks:
            section = (
                chunk.metadata.get("Header 1")
                or chunk.metadata.get("Header 2")
                or chunk.metadata.get("Header 3")
            )
            chunk.metadata = {
                "chunk_id":  str(uuid.uuid4()),
                "parent_id": parent_id,
                "source":    source,
                "page":      page,
                "section":   section,
                "file_type": file_type,
                "origin":    doc.metadata.get("origin", "local"),
                "user_id":   doc.metadata.get("user_id", "public"),
            }
            standardized_chunks.append(chunk)

    return standardized_chunks
