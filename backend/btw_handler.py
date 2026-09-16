"""
Handles /btw side-channel questions — explicit off-topic queries the user
prefixes with /btw. These are NOT saved to session history, and NOT routed
through the main paper-focused graph.
"""
from dotenv import load_dotenv
load_dotenv()

import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

llm = ChatOpenAI(model="gpt-5-mini")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


class BtwDecision(BaseModel):
    """Decides whether a /btw question needs a live web search or can be
    answered directly from the model's own knowledge."""
    needs_web_search: bool = Field(
        description="True if the question asks about current events, recent "
                    "developments, or anything requiring up-to-date information. "
                    "False for general knowledge, definitions, or timeless facts."
    )


btw_decision_llm = llm.with_structured_output(BtwDecision)


def is_btw_command(message: str) -> bool:
    """Checks if a message uses the /btw prefix."""
    return message.strip().lower().startswith("/btw")


def handle_btw(message: str) -> str:
    """
    Handles a /btw-prefixed message: strips the prefix, decides whether to
    answer directly or search the web, and returns the answer. Callers should
    NOT save this exchange to session history.
    """
    question = message.strip()[4:].strip()  # remove "/btw" prefix

    if not question:
        return "Usage: `/btw <your question>` — ask anything outside the current paper's context."

    decision = btw_decision_llm.invoke(question)

    if decision.needs_web_search:
        search_results = tavily_client.search(query=question, max_results=3)
        results_text = "\n\n".join(
            f"Source: {r['url']}\n{r['content']}"
            for r in search_results.get("results", [])
        )
        prompt = f"""
            Answer the question using the web search results below. Be concise.

            Search results:
            {results_text}

            Question: {question}
            """
        response = llm.invoke(prompt)
    else:
        response = llm.invoke(question)

    return response.content