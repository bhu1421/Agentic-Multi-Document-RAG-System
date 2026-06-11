import time
import json
from langchain_core.prompts import ChatPromptTemplate


def query_expansion_node(state):
    """Generate multiple search variants for each planner task to boost recall.

    Takes the planner's task list and produces 3 diverse reformulations per task
    using different keywords, synonyms, and angles. The original tasks are always
    preserved alongside the expansions so we never lose the user's intent.
    """
    from backend.llm import get_llm

    llm = get_llm()
    tasks = state["tasks"]

    expansion_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a search query expansion expert. Given a search task, generate exactly 3 diverse "
         "search query variants that capture different phrasings, synonyms, and angles of the same "
         "information need.\n\n"
         "Rules:\n"
         "- Each variant should use different keywords and phrases\n"
         "- Cover synonyms, related terms, and alternative phrasings\n"
         "- Keep each variant concise (under 15 words)\n"
         "- Do NOT repeat the original task verbatim\n"
         "- Output ONLY a JSON list of 3 strings, no markdown or extra text\n\n"
         "Examples:\n"
         "Task: 'AWS security permissions'\n"
         "Output: [\"AWS IAM policies and roles\", \"AWS access control permissions management\", "
         "\"Amazon web services security authorization\"]\n\n"
         "Task: 'machine learning algorithms'\n"
         "Output: [\"ML models and techniques\", \"supervised unsupervised learning methods\", "
         "\"artificial intelligence algorithm types\"]"),
        ("human", "Task: {task}")
    ])

    all_expanded = []

    for task in tasks:
        t = time.time()
        try:
            response = (expansion_prompt | llm).invoke({"task": task}).content.strip()

            # Strip markdown fences if the LLM wrapped them
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "").strip()
            elif response.startswith("```"):
                response = response.replace("```", "").strip()

            variants = json.loads(response)
            if isinstance(variants, list) and len(variants) > 0:
                all_expanded.extend([v for v in variants if isinstance(v, str)])
                print(f"[QueryExpansion] '{task}' -> {variants} ({time.time() - t:.1f}s)")
            else:
                print(f"[QueryExpansion] Unexpected format for '{task}', keeping original")
        except Exception as e:
            print(f"[QueryExpansion] Failed to expand '{task}': {e}")

    # Always include original tasks — never lose the user's own wording
    combined = list(dict.fromkeys(tasks + all_expanded))  # dedup preserving order

    print(f"[QueryExpansion] Total: {len(tasks)} tasks -> {len(combined)} expanded queries")
    return {"expanded_queries": combined}
