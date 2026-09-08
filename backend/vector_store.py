from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

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
    Retrieves the top-k most relevant chunks for a query from the session's collection.
    """
    vector_store = get_vector_store(session_id)
    return vector_store.similarity_search(query, k=k)