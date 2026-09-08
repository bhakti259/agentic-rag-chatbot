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