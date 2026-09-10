from pydantic import BaseModel, Field
from typing import Literal


class RouteDecision(BaseModel):
    """
    Structured output for the router node — decides how to handle a user's query.
    """
    route: Literal["direct_answer", "retrieve", "web_search", "verify_claim"] = Field(
        description="Which path the query should take: "
                    "'direct_answer' for general/timeless knowledge questions answerable without any paper or web search — e.g. facts, definitions, explanations that don't change over time. "
                    "'retrieve' for questions about the specific content of an uploaded paper — e.g. 'summarize', 'what methodology', 'what did the authors conclude'. "
                    "'web_search' for questions about current events, recent developments, or 'the latest' research on a general topic — NOT tied to verifying a specific claim from a paper. "
                    "'verify_claim' for questions asking whether a specific claim, finding, or statement from a paper is still true, current, or has been superseded — look for phrasing like 'is it still true that...', 'has this been disproven...', 'has this finding held up...'"
    )
    reasoning: str = Field(
        description="Brief explanation of why this route was chosen."
    )
    
class RelevancyCheck(BaseModel):
    """
    Structured output for checking whether retrieved chunks are relevant
    enough to answer the user's query.
    """
    is_relevant: bool = Field(
        description="True if the retrieved chunks contain the core information needed to give "
                    "a reasonable, grounded answer to the query — even if not exhaustive or missing "
                    "minor details. False only if the chunks are off-topic, about a different subject "
                    "entirely, or missing the key fact(s) the query is actually asking for."
    )
    reasoning: str = Field(
        description="Brief explanation of why the chunks were judged relevant or not."
    )
    
class QueryRewrite(BaseModel):
    """
    Structured output for rewriting a query that failed the relevancy check,
    to improve retrieval on the next attempt.
    """
    rewritten_query: str = Field(
        description="A revised version of the original query, rephrased to better "
                    "match likely terminology in the paper (e.g. more specific, "
                    "different synonyms, or breaking a compound question into its core part)."
    )
    
class VerdictResult(BaseModel):
    """
    Structured output for the claim verification step — summarizes whether
    a claim from the paper still holds up against current web/arXiv findings.
    """
    verdict: Literal["still_valid", "outdated", "contradicted", "inconclusive"] = Field(
    description="'still_valid' if current sources support the claim unchanged. "
                "'outdated' if the claim was accurate when made, but has since been superseded by a better "
                "approach or newer consensus — the original claim wasn't wrong, it's just no longer the "
                "current best answer (e.g., 'RNNs were best for translation' → later surpassed by Transformers "
                "is OUTDATED, not contradicted, since RNNs really were effective at the time). "
                "'contradicted' if newer findings show the original claim was actually incorrect or its "
                "reasoning flawed — not merely 'a better option came along', but 'this was never really true' "
                "or 'this stopped working the way it was claimed to'. "
                "'inconclusive' if search results don't provide enough evidence either way."
    )
    explanation: str = Field(
        description="A clear explanation of the verdict, referencing what the search results showed."
    )
    supporting_sources: list[str] = Field(
        description="URLs of the most relevant sources found (web articles or arXiv papers) that inform this verdict.",
        default_factory=list,
    )
    

class OffTopicCheck(BaseModel):
    """
    Structured output for detecting whether a query is genuinely off-topic
    (chit-chat, unrelated requests) rather than a real research question.
    """
    is_off_topic: bool = Field(
        description="True if the query is casual conversation, a greeting, or "
                    "completely unrelated to research papers/analysis (e.g. 'how are you', "
                    "'tell me a joke', 'what's 2+2'). False for any genuine question about "
                    "research, papers, or claims — even if informally phrased."
    )