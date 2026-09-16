"""
Evaluation pipeline for the agentic RAG chatbot, using DeepEval to measure
retrieval and generation quality against a hand-written golden dataset.
"""
import json
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
from backend.paper_loader import load_arxiv_by_id


from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from backend.golden_dataset import RAG_GOLDEN_DATASET, ROUTING_GOLDEN_DATASET, GOLDEN_DATASET
from backend.rag_graph import app as rag_app
from backend.paper_loader import load_paper
from backend.vector_store import add_chunks

EVAL_SESSION_ID = "evaluation_session"
THRESHOLD = 0.7

def setup_eval_session():
    """
    Loads the GPT-4 System Card into a dedicated, freshly-cleared evaluation
    session — evaluation must be deterministic/reproducible, unlike a real
    user session where documents intentionally accumulate.
    """
    collection_name = f"papeer_{EVAL_SESSION_ID}"
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    chunks = load_arxiv_by_id("2303.08774")  # deterministic, bypasses search ambiguity
    add_chunks(EVAL_SESSION_ID, chunks)
    
def run_pipeline_for_question(question: str):
    """
    Runs a single question through the compiled graph and returns
    (actual_answer, retrieved_context_list) for DeepEval scoring.
    """
    result = rag_app.invoke({
        "messages": [],
        "session_id": EVAL_SESSION_ID,
        "original_query": question,
        "query": question,
        "route": "",
        "retrieved_chunks": [],
        "is_relevant": False,
        "rewrite_count": 0,
        "verdict": "",
        "final_answer": "",
    })

    actual_answer = result["final_answer"]
    retrieved_context = [doc.page_content for doc in result["retrieved_chunks"]]

    return actual_answer, retrieved_context

def build_test_cases(dataset):
    test_cases = []
    for item in dataset:
        question = item["question"]
        expected_answer = item["expected_answer"]

        print(f"Running: {question}")
        actual_answer, retrieved_context = run_pipeline_for_question(question)

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_answer,
            expected_output=expected_answer,
            retrieval_context=retrieved_context,
        )
        test_cases.append(test_case)

    return test_cases

def run_evaluation():
    setup_eval_session()

    print("\n=== Evaluating RAG-grounded questions (all 5 metrics) ===")
    rag_test_cases = build_test_cases(RAG_GOLDEN_DATASET[:2])  # limit to first 2 for brevity
    rag_metrics = [
        ContextualPrecisionMetric(threshold=THRESHOLD),
        ContextualRecallMetric(threshold=THRESHOLD),
        ContextualRelevancyMetric(threshold=THRESHOLD),
        AnswerRelevancyMetric(threshold=THRESHOLD),
        FaithfulnessMetric(threshold=THRESHOLD),
    ]
    evaluate(test_cases=rag_test_cases, metrics=rag_metrics)

    print("\n=== Evaluating routing-correctness questions (Answer Relevancy only) ===")
    routing_test_cases = build_test_cases(ROUTING_GOLDEN_DATASET)
    routing_metrics = [AnswerRelevancyMetric(threshold=THRESHOLD)]
    evaluate(test_cases=routing_test_cases, metrics=routing_metrics)


if __name__ == "__main__":
    run_evaluation()