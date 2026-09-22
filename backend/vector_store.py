from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import re

load_dotenv()

def get_embeddings():
    """
    Returns an OpenAI embeddings model wrapped with a local disk cache,
    so identical text is never re-embedded (saves API calls + latency).
    """
    underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    store = LocalFileStore("./embedding_cache/")

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        store,
        namespace=underlying_embeddings.model,
        key_encoder="sha256",

    )

    return cached_embeddings


def get_vector_store(session_id: str):
    """
    Returns a Qdrant vector store scoped to a specific session.
    Creates the collection if it doesn't already exist.

    Args:
        session_id: unique identifier for the user's session

    Returns:
        a QdrantVectorStore instance tied to a session-specific collection
    """
    collection_name = f"papeer_{session_id}"

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=get_embeddings(),
    )

    return vector_store


def add_chunks(session_id: str, chunks: list):
    """
    Embeds and stores a list of Document chunks in the session's collection.
    """
    vector_store = get_vector_store(session_id)
    vector_store.add_documents(chunks)


def retrieve(session_id: str, query: str, k: int = 4):
    """
    Retrieves the top-k most relevant chunks for a query from the session's
    collection, filtering out citation-heavy chunks that dilute relevancy.
    """
    vector_store = get_vector_store(session_id)

    # Retrieve extra candidates so we still have k good ones after filtering
    candidates = vector_store.similarity_search(query, k=k * 3)

    filtered = [doc for doc in candidates if not is_citation_heavy(doc.page_content)]

    return filtered[:k] if filtered else candidates[:k]  # fallback if all filtered

def is_citation_heavy(text: str, threshold: float = 0.3) -> bool:
    """
    Heuristic check for whether a chunk is predominantly citations/references/
    acknowledgments rather than substantive content. Used to filter out noise
    that dilutes retrieval relevancy (see eval findings: GPT-4 System Card's
    large bibliography/acknowledgments sections were polluting retrieval).
    """
    if not text.strip():
        return True

    # Count citation-like patterns: "Author, A.", "et al.", "(YYYY)"
    citation_patterns = re.findall(
        r'[A-Z][a-z]+,\s[A-Z]\.|\bet al\.|\(\d{4}\)',
        text
    )

    # Count comma-separated name-like fragments (common in acknowledgment lists)
    words = text.split()
    if not words:
        return True

    citation_signal_density = len(citation_patterns) / max(len(words) / 20, 1)

    return citation_signal_density > threshold