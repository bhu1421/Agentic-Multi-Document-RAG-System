import uuid
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    Language
)

def chunk_documents(documents):
    """Split documents and enforce the Metadata Standard using context-aware chunking."""
    standardized_chunks = []
    
    # Base fallback splitter
    base_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    
    md_headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=md_headers_to_split_on)
    
    html_headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=html_headers_to_split_on)
    
    language_map = {
        "py": Language.PYTHON,
        "js": Language.JS,
        "ts": Language.TS,
        "go": Language.GO,
        "java": Language.JAVA,
        "cpp": Language.CPP,
        "c": Language.C,
        "cs": Language.CSHARP,
        "rb": Language.RUBY,
        "rs": Language.RUST,
        "php": Language.PHP
    }
    
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)
        
        if source.startswith("http"):
            file_type = "web"
        elif ".git" in source or "/cloned_repos/" in source:
            file_type = source.split(".")[-1].lower() if "." in source else "unknown"
        else:
            file_type = source.split(".")[-1].lower() if "." in source else "unknown"
            
        parent_id = str(uuid.uuid4())
        chunks = []
        
        if file_type == "pdf":
            chunks = base_splitter.split_documents([doc])
        elif file_type == "md":
            md_docs = md_splitter.split_text(doc.page_content)
            chunks = base_splitter.split_documents(md_docs)
            # Retain original metadata
            for c in chunks:
                c.metadata.update(doc.metadata)
        elif file_type in ["html", "web"]:
            try:
                html_docs = html_splitter.split_text(doc.page_content)
                chunks = base_splitter.split_documents(html_docs)
                for c in chunks:
                    c.metadata.update(doc.metadata)
            except Exception:
                chunks = base_splitter.split_documents([doc])
        elif file_type == "csv":
            # CSV is already one row per document
            chunks = [doc]
        elif file_type in language_map:
            code_splitter = RecursiveCharacterTextSplitter.from_language(
                language=language_map[file_type], chunk_size=800, chunk_overlap=150
            )
            chunks = code_splitter.split_documents([doc])
        else:
            chunks = base_splitter.split_documents([doc])
            
        for chunk in chunks:
            # Determine section from headers if available
            section = None
            if "Header 1" in chunk.metadata:
                section = chunk.metadata["Header 1"]
            elif "Header 2" in chunk.metadata:
                section = chunk.metadata["Header 2"]
            elif "Header 3" in chunk.metadata:
                section = chunk.metadata["Header 3"]
                
            chunk_id = str(uuid.uuid4())
            
            chunk.metadata = {
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "source": source,
                "page": page,
                "section": section,
                "file_type": file_type
            }
            standardized_chunks.append(chunk)

    return standardized_chunks
