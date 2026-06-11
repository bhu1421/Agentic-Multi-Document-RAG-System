import time
import json
from langchain_core.prompts import ChatPromptTemplate


def metadata_filter_node(state):
    """Extract structured metadata filters from the user's natural-language query.

    Analyzes the query to detect explicit references to document metadata fields
    (source filename, file type, page number, section/header name) and builds a
    filter dict that the Retriever can translate into Qdrant FieldConditions.

    Only extracts filters that are **explicitly stated** — never guesses.
    Returns an empty dict if no filterable metadata is detected.
    """
    from backend.llm import get_llm

    llm = get_llm()
    query = state["query"]

    filter_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a metadata extraction expert for a document search system. "
         "Analyze the user's query and extract any filter conditions that refer to "
         "specific document metadata.\n\n"
         "Available metadata fields:\n"
         "- source: The filename (e.g., 'report.pdf', 'main.py', 'notes.md')\n"
         "- file_type: The type of file. Valid values: pdf, txt, md, py, js, ts, "
         "go, java, cpp, c, cs, rb, rs, php, docx, csv, xlsx, pptx, web\n"
         "- page: The page number as an integer (only meaningful for PDFs). "
         "Use 0-indexed numbering: user's 'page 1' = 0, 'page 5' = 4\n"
         "- section: The section or header name (e.g., 'Introduction', 'Methods', "
         "'Conclusion')\n\n"
         "Rules:\n"
         "- Only extract filters EXPLICITLY mentioned in the query\n"
         "- Do NOT guess or infer filters that aren't clearly stated\n"
         "- If no filters are detectable, output an empty JSON object: {}\n"
         "- Output ONLY valid JSON, no markdown fences or extra text\n\n"
         "Examples:\n"
         "Query: 'What does page 5 of report.pdf say?' -> "
         '{{"source": "report.pdf", "page": 4}}\n'
         "Query: 'Show me only Python files' -> "
         '{{"file_type": "py"}}\n'
         "Query: 'What is in the Introduction section?' -> "
         '{{"section": "Introduction"}}\n'
         "Query: 'Summarize the conclusion of thesis.pdf' -> "
         '{{"source": "thesis.pdf", "section": "Conclusion"}}\n'
         "Query: 'Tell me about machine learning' -> {{}}\n"
         "Query: 'Compare all my documents' -> {{}}"),
        ("human", "Query: {query}")
    ])

    t = time.time()
    try:
        response = (filter_prompt | llm).invoke({"query": query}).content.strip()

        # Strip markdown fences
        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "").strip()
        elif response.startswith("```"):
            response = response.replace("```", "").strip()

        filters = json.loads(response)

        if not isinstance(filters, dict):
            filters = {}

        # Type-safety: ensure page is int if present
        if "page" in filters:
            try:
                filters["page"] = int(filters["page"])
            except (ValueError, TypeError):
                del filters["page"]

        # Remove null / empty values
        filters = {k: v for k, v in filters.items() if v is not None and v != ""}

    except Exception as e:
        print(f"[MetadataFilter] Failed to parse filters: {e}")
        filters = {}

    print(f"[MetadataFilter] Query: '{query}' -> Filters: {filters} ({time.time() - t:.1f}s)")
    return {"metadata_filters": filters}
