import streamlit as st
import uuid
import tempfile
import os
from backend.rag_graph import app as rag_app
from backend.btw_handler import handle_off_topic
from backend.paper_loader import load_paper
from backend.vector_store import add_chunks

st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="📄")
st.title("📄 Agentic RAG Chatbot")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "paper_loaded" not in st.session_state:
    st.session_state.paper_loaded = False

# --- Sidebar: paper ingestion ---
with st.sidebar:
    st.header("📥 Load a Paper")

    input_type = st.radio(
        "Choose input type:",
        ["Upload File (PDF/TXT/MD)", "Web URL", "ArXiv"],
    )

    if input_type == "Upload File (PDF/TXT/MD)":
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])

        if uploaded_file and st.button("Load File"):
            with st.spinner("Processing document..."):
                # Save uploaded file to a temp path so our loaders (which expect file paths) can read it
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                source_type = "pdf" if suffix == ".pdf" else "text"

                try:
                    chunks = load_paper(tmp_path, source_type)
                    add_chunks(st.session_state.session_id, chunks)
                    st.session_state.paper_loaded = True
                    st.success(f"Loaded {len(chunks)} chunks from {uploaded_file.name}")
                finally:
                    os.remove(tmp_path)

    elif input_type == "Web URL":
        url = st.text_input("Enter a URL:")

        if url and st.button("Load URL"):
            with st.spinner("Fetching and processing page..."):
                chunks = load_paper(url, "url")
                add_chunks(st.session_state.session_id, chunks)
                st.session_state.paper_loaded = True
                st.success(f"Loaded {len(chunks)} chunks from URL")

    elif input_type == "ArXiv":
        arxiv_query = st.text_input("Enter a paper title or ArXiv ID:")

        if arxiv_query and st.button("Load from ArXiv"):
            with st.spinner("Searching ArXiv and processing paper..."):
                chunks = load_paper(arxiv_query, "arxiv")
                add_chunks(st.session_state.session_id, chunks)
                st.session_state.paper_loaded = True
                st.success(f"Loaded {len(chunks)} chunks from ArXiv")

    if st.session_state.paper_loaded:
        st.info("✅ A paper is loaded for this session.")


# --- Main chat area (unchanged from before) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question about your paper...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    off_topic_response = handle_off_topic(user_query)

    if off_topic_response:
        answer = off_topic_response
    else:
        result = rag_app.invoke({
            "messages": [],
            "session_id": st.session_state.session_id,
            "query": user_query,
            "route": "",
            "retrieved_chunks": [],
            "is_relevant": False,
            "rewrite_count": 0,
            "verdict": "",
            "final_answer": "",
        })
        answer = result["final_answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)