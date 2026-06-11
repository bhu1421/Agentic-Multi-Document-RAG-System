import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from backend.llm import get_llm
from backend.vectordb import get_embeddings

def evaluate_interaction(question: str, answer: str, contexts: list, ground_truth: str = None):
    """
    Evaluate a single RAG interaction using Ragas.
    Since Context Precision and Recall require a ground truth, if none is provided,
    we use the generated answer as a proxy to allow the metrics to compute, 
    though this is for demonstration purposes.
    """
    if not ground_truth:
        ground_truth = answer
        
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [[doc.page_content for doc in contexts]],
        "ground_truth": [ground_truth]
    }
    
    dataset = Dataset.from_dict(data)
    
    llm = get_llm()
    embeddings = get_embeddings()
    
    # Ragas needs specific wrappers for custom Langchain models depending on the version,
    # but the generic evaluate function accepts langchain models directly in recent versions.
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
    
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False
        )
        return result
    except Exception as e:
        print(f"[Evaluation Error] {e}")
        return None
