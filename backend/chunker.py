import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    """Split documents and enforce the Metadata Standard."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    
    raw_chunks = splitter.split_documents(documents)
    standardized_chunks = []
    
    for chunk in raw_chunks:
        # Extract existing metadata
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", None)
        
        # Determine file type
        if source.startswith("http"):
            file_type = "web"
        elif ".git" in source or "/cloned_repos/" in source:
            file_type = "github"
        else:
            file_type = source.split(".")[-1].lower() if "." in source else "unknown"
            
        # Enforce Metadata Standard
        chunk.metadata = {
            "document_id": str(uuid.uuid4()),
            "source": source,
            "page": page,
            "section": None,
            "file_type": file_type
        }
        
        standardized_chunks.append(chunk)
        
    return standardized_chunks
