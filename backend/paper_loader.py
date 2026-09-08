from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

import os
import tempfile
import requests
from tavily import TavilyClient

from dotenv import load_dotenv
load_dotenv()


def load_pdf(file_path: str):
    """
    Load a PDF file and split it into chunks for embedding.

    Args:
        file_path: path to the PDF file on disk

    Returns:
        list of LangChain Document objects, each representing one chunk
    """
    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()  # one Document per page

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(raw_documents)

    return chunks


def load_text(file_path: str):
    """
    Load a TXT or MD file and split it into chunks.

    Args:
        file_path: path to the .txt or .md file on disk

    Returns:
        list of LangChain Document objects, each representing one chunk
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.create_documents(
        texts=[raw_text],
        metadatas=[{"source": file_path}],
    )

    return chunks



def load_urls(urls: list[str]):
    """
    Load one or more web pages and split them into chunks.

    Args:
        urls: list of web page URLs

    Returns:
        list of LangChain Document objects, each representing one chunk
    """
    loader = WebBaseLoader(urls)
    raw_documents = loader.load()  # one Document per URL

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(raw_documents)

    return chunks


def load_arxiv(query: str):
    """
    Search ArXiv for a paper (by title or ID) and load it as chunks.

    Args:
        query: paper title or ArXiv ID (e.g. "2303.08774")

    Returns:
        list of LangChain Document objects, each representing one chunk
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    # Step 1: find the paper's arXiv page via Tavily, restricted to arxiv.org
    search_results = tavily_client.search(
        query=f"{query} site:arxiv.org"
    )
    results = search_results.get("results", [])
    if not results:
        raise ValueError(f"No arXiv paper found for query: {query}")

    paper_url = results[0]["url"]  # e.g. https://arxiv.org/abs/2303.08774

    # Step 2: arXiv abstract pages have a matching PDF at a predictable URL
    arxiv_id = paper_url.rstrip("/").split("/")[-1]
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    # Step 3: download the PDF to a temp file, then reuse our existing loader
    response = requests.get(pdf_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    try:
        chunks = load_pdf(tmp_path)
    finally:
        os.remove(tmp_path)  # always clean up, even if load_pdf fails

    return chunks


def load_paper(source: str, source_type: str):
    """
    Unified entry point — dispatches to the correct loader based on source_type.

    Args:
        source: file path, URL, or arXiv query/ID (single string)
        source_type: one of "pdf", "text", "url", "arxiv"

    Returns:
        list of LangChain Document chunks
    """
    if source_type == "pdf":
        return load_pdf(source)
    elif source_type == "text":
        return load_text(source)
    elif source_type == "url":
        return load_urls([source])
    elif source_type == "arxiv":
        return load_arxiv(source)
    else:
        raise ValueError(f"Unknown source_type: {source_type}")