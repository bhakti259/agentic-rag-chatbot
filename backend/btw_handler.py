from backend.models import OffTopicCheck
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()


llm = ChatOpenAI(model="gpt-5-mini")
offtopic_llm = llm.with_structured_output(OffTopicCheck)


def handle_off_topic(query: str) -> str | None:
    """
    Checks if a query is off-topic. If so, returns a friendly redirect message.
    If not, returns None (signaling the query should proceed to the main graph).
    """
    result = offtopic_llm.invoke(query)

    if result.is_off_topic:
        return (
            "I'm a research paper assistant — I can help you analyze uploaded papers, "
            "search for current research, or verify claims against recent findings. "
            "What paper or topic would you like to explore?"
        )

    return None