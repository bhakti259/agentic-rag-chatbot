from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from backend.models import RouteDecision
from backend.vector_store import retrieve as vector_retrieve
from backend.models import RelevancyCheck, QueryRewrite
from langgraph.graph import StateGraph, END


class GraphState(TypedDict):
    """
    Shared state passed between all nodes in the graph.
    """
    messages: Annotated[list, add_messages]  # conversation history
    session_id: str
    query: str                    # current user query (may get rewritten)
    route: str                    # decision from the router
    retrieved_chunks: list        # chunks from vector store
    is_relevant: bool             # relevancy check result
    rewrite_count: int            # tracks retry attempts (cap at 3)
    verdict: str                  # claim verification result
    final_answer: str             # what gets shown to the user
    
  
llm = ChatOpenAI(model="gpt-5-mini")
router_llm = llm.with_structured_output(RouteDecision)


def router_node(state: GraphState) -> dict:
    """
    Classifies the user's query into one of four routes.
    """
    query = state["query"]
    decision = router_llm.invoke(query)

    return {
        "route": decision.route,
    }
    
def retrieve_node(state: GraphState) -> dict:
    """
    Retrieves relevant chunks from the vector store for the current query.
    """
    chunks = vector_retrieve(
        session_id=state["session_id"],
        query=state["query"],
        k=4,
    )

    return {
        "retrieved_chunks": chunks,
    }
    

relevancy_llm = llm.with_structured_output(RelevancyCheck)


def relevancy_check_node(state: GraphState) -> dict:
    """
    Checks whether the retrieved chunks are good enough to answer the query.
    """
    chunks_text = "\n\n".join(doc.page_content for doc in state["retrieved_chunks"])

    prompt = f"""
        User query: {state["query"]}

        Retrieved chunks:
        {chunks_text}

        Are these chunks relevant enough to answer the query?
    """

    result = relevancy_llm.invoke(prompt)

    return {
        "is_relevant": result.is_relevant,
    }
    

rewrite_llm = llm.with_structured_output(QueryRewrite)


def rewrite_node(state: GraphState) -> dict:
    """
    Rewrites the query when retrieval failed the relevancy check.
    Increments the rewrite counter to enforce the retry cap.
    """
    prompt = f"""
        Original query: {state["query"]}

        This query failed to retrieve relevant chunks from the document.

        Suggest a rewritten query that might retrieve better results.
    """

    result = rewrite_llm.invoke(prompt)

    return {
        "query": result.rewritten_query,
        "rewrite_count": state["rewrite_count"] + 1,
    }
    
def generate_node(state: GraphState) -> dict:
    """
    Generates the final answer using retrieved chunks as context.
    """
    chunks_text = "\n\n".join(doc.page_content for doc in state["retrieved_chunks"])

    prompt = f"""
        Answer the user's question using only the context below. If the context
        doesn't contain enough information, say so honestly rather than guessing.

        Context:
        {chunks_text}

        Question: {state["query"]}
    """

    response = llm.invoke(prompt)

    return {
        "final_answer": response.content,
    }
    
    
def should_rewrite(state: GraphState) -> str:
    """
    Conditional edge: decides whether to retry retrieval or give up and generate.
    """
    if state["is_relevant"]:
        return "generate"
    elif state["rewrite_count"] >= 3:
        return "generate"  # give up after 3 attempts, answer with what we have
    else:
        return "rewrite"


graph = StateGraph(GraphState)

graph.add_node("router", router_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("relevancy_check", relevancy_check_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("router")
graph.add_edge("router", "retrieve")
graph.add_edge("retrieve", "relevancy_check")

graph.add_conditional_edges(
    "relevancy_check",
    should_rewrite,
    {
        "rewrite": "rewrite",
        "generate": "generate",
    },
)

graph.add_edge("rewrite", "retrieve")  # loop back after rewriting
graph.add_edge("generate", END)

app = graph.compile()
