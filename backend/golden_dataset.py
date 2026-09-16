"""
Golden dataset for evaluating the RAG pipeline's retrieval and generation
quality using DeepEval. Based on the GPT-4 System Card (arXiv 2303.08774).

Questions are explicitly anchored ("according to this paper", "based on this
document") to force the router toward the `retrieve` path — otherwise the
router may (correctly, per its own logic) route a generally-answerable
question to `direct_answer`, skipping retrieval entirely and making
context-based metrics (Contextual Precision/Recall/Relevancy) meaningless
(they score 0 when no context was retrieved at all, not because retrieval
was bad).

The last two entries are deliberately NOT document-anchored — they test
routing correctness (direct_answer / web_search) rather than RAG retrieval
quality, and should be evaluated with Answer Relevancy only, not the
context-based metrics.
"""

# Document-grounded questions — should route to `retrieve`
RAG_GOLDEN_DATASET = [
    {
        "question": "According to this paper, what is the main topic being discussed?",
        "expected_answer": (
            "The paper discusses the safety challenges, evaluation methods, and "
            "deployment mitigations for GPT-4, including risks like disinformation, "
            "adversarial testing (red-teaming), and recommendations for safe deployment."
        ),
    },
    {
        "question": "Based on this paper, what is one of the safety risks discussed regarding GPT-4?",
        "expected_answer": (
            "One safety risk discussed is disinformation — GPT-4's potential to "
            "generate convincing false or misleading content at scale."
        ),
    },
    {
        "question": "According to this document, what is red-teaming?",
        "expected_answer": (
            "Red-teaming refers to adversarial testing conducted on GPT-4 before "
            "deployment, where testers attempted to elicit harmful, unsafe, or "
            "unintended behavior from the model to identify risks."
        ),
    },
    {
        "question": "Does this paper describe the model's training methodology, such as dataset sizes or architecture details?",
        "expected_answer": (
            "No — the System Card explicitly does not disclose detailed training "
            "methodology, architecture specifics, dataset composition, or hardware "
            "used, citing competitive and safety considerations."
        ),
    },
    {
        "question": "According to the acknowledgments in this paper, what organization was involved as a partner?",
        "expected_answer": (
            "Microsoft is acknowledged as a partner, particularly in relation to "
            "deployment support."
        ),
    },
    {
        "question": "In this paper, is InstructGPT described as this document's own main contribution?",
        "expected_answer": (
            "No — InstructGPT is a different paper, only referenced within this "
            "document as an example in a multimodal capability demonstration "
            "(GPT-4 summarizing a photo of the InstructGPT paper). It is not this "
            "document's own contribution, and the paper's own main contribution is "
            "about GPT-4's safety evaluation and deployment, not InstructGPT."
        ),
    },
]

# Routing-correctness questions — deliberately NOT document-anchored.
# Evaluate with Answer Relevancy only; context metrics don't apply since
# no retrieval should occur for these.
ROUTING_GOLDEN_DATASET = [
    {
        "question": "What is the capital of France?",
        "expected_answer": (
            "Paris. (Tests that the system correctly routes to direct_answer "
            "rather than attempting document retrieval.)"
        ),
    },
    {
        "question": "What are the latest developments in AI safety research in 2026?",
        "expected_answer": (
            "This should trigger a live web search rather than being answered "
            "from the loaded document, since it asks about current developments "
            "beyond what's in the paper."
        ),
    },
]

# Combined, for convenience where a single flat list is useful
GOLDEN_DATASET = RAG_GOLDEN_DATASET + ROUTING_GOLDEN_DATASET