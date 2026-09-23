import streamlit as st
import uuid
import tempfile
import os
from backend.rag_graph import app as rag_app
from backend.btw_handler import is_btw_command, handle_btw
from backend.paper_loader import load_paper
from backend.vector_store import add_chunks
import json
from pathlib import Path


SESSIONS_FILE = Path("sessions.json")

def load_sessions_metadata() -> dict:
    """Loads session metadata (id -> {title, papers_loaded}) from disk."""
    if SESSIONS_FILE.exists():
        with open(SESSIONS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_sessions_metadata(sessions_metadata: dict):
    """Persists session metadata to disk."""
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions_metadata, f, indent=2)

st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="📄")

# --- Multi-session state: a dict of session_id -> session data ---
if "sessions" not in st.session_state:
    persisted = load_sessions_metadata()
    st.session_state.sessions = {
        sid: {"messages": [], "display_messages": [], "papers_loaded": meta["papers_loaded"], "title": meta["title"]}
        for sid, meta in persisted.items()
    }

if "active_session_id" not in st.session_state:
    if st.session_state.sessions:
        # Resume the most recently used session
        st.session_state.active_session_id = list(st.session_state.sessions.keys())[-1]
    else:
        new_id = str(uuid.uuid4())
        st.session_state.sessions[new_id] = {
            "messages": [], "display_messages": [], "papers_loaded": [], "title": "New Chat"
        }
        st.session_state.active_session_id = new_id
        save_sessions_metadata({
            sid: {"title": s["title"], "papers_loaded": s["papers_loaded"]}
            for sid, s in st.session_state.sessions.items()
        })

def create_new_session():
    new_id = str(uuid.uuid4())
    st.session_state.sessions[new_id] = {
        "messages": [], "display_messages": [], "papers_loaded": [], "title": "New Chat"
    }
    st.session_state.active_session_id = new_id
    save_sessions_metadata({
        sid: {"title": s["title"], "papers_loaded": s["papers_loaded"]}
        for sid, s in st.session_state.sessions.items()
    })


# --- Sidebar: session management ---
with st.sidebar:
    st.header("💬 Sessions")

    if st.button("➕ New Chat"):
        create_new_session()
        st.rerun()

    st.divider()

    for sid, sdata in st.session_state.sessions.items():
        label = sdata["title"]
        if sdata["papers_loaded"]:
            label += f" ({len(sdata['papers_loaded'])} papers)"
        if st.button(label, key=f"switch_{sid}"):
            st.session_state.active_session_id = sid
            st.rerun()

    st.divider()

    # --- Paper ingestion (scoped to the ACTIVE session) ---
    st.header("📥 Load a Paper")

    active_id = st.session_state.active_session_id
    active_session = st.session_state.sessions[active_id]

    input_type = st.radio(
        "Choose input type:",
        ["Upload File (PDF/TXT/MD)", "Web URL", "ArXiv"],
    )

    if input_type == "Upload File (PDF/TXT/MD)":
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])

        if uploaded_file and st.button("Load File"):
            with st.spinner("Processing document..."):
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                source_type = "pdf" if suffix == ".pdf" else "text"

                try:
                    chunks = load_paper(tmp_path, source_type)
                    add_chunks(active_id, chunks)
                    active_session["papers_loaded"].append(uploaded_file.name)
                    
                    save_sessions_metadata({
                    sid: {"title": s["title"], "papers_loaded": s["papers_loaded"]}
                    for sid, s in st.session_state.sessions.items()
                    })
                    st.success(f"Loaded {len(chunks)} chunks from {uploaded_file.name}")
                finally:
                    os.remove(tmp_path)

    elif input_type == "Web URL":
        url = st.text_input("Enter a URL:")
        if url and st.button("Load URL"):
            with st.spinner("Fetching and processing page..."):
                try:
                    chunks = load_paper(url, "url")
                    add_chunks(active_id, chunks)
                    active_session["papers_loaded"].append(url)
                    save_sessions_metadata({
                    sid: {"title": s["title"], "papers_loaded": s["papers_loaded"]}
                    for sid, s in st.session_state.sessions.items()
                    })
                    st.success(f"Loaded {len(chunks)} chunks from URL")
                except ValueError as e:
                    st.error(str(e))

    elif input_type == "ArXiv":
        arxiv_query = st.text_input("Enter a paper title or ArXiv ID:")
        if arxiv_query and st.button("Load from ArXiv"):
            with st.spinner("Searching ArXiv and processing paper..."):
                chunks = load_paper(arxiv_query, "arxiv")
                add_chunks(active_id, chunks)
                active_session["papers_loaded"].append(arxiv_query)
                save_sessions_metadata({
                sid: {"title": s["title"], "papers_loaded": s["papers_loaded"]}
                for sid, s in st.session_state.sessions.items()
                })
                st.success(f"Loaded {len(chunks)} chunks from ArXiv")

    if active_session["papers_loaded"]:
        st.info(f"📄 Loaded papers: {', '.join(active_session['papers_loaded'])}")


# --- Main chat area (scoped to the active session) ---
st.title(f"📄 Agentic RAG Chatbot")
st.caption(f"Session: {active_id[:8]}")

for message in active_session["display_messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question about your paper...")

if user_query:
    if is_btw_command(user_query):
        active_session["display_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        answer = handle_btw(user_query)

        active_session["display_messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        active_session["messages"].append({"role": "user", "content": user_query})
        active_session["display_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        result = rag_app.invoke({
            "messages": [],
            "session_id": active_id,
            "original_query": user_query,
            "query": user_query,
            "route": "",
            "retrieved_chunks": [],
            "is_relevant": False,
            "rewrite_count": 0,
            "verdict": "",
            "final_answer": "",
        },
        config={"configurable": {"thread_id": active_id}}
        )
        answer = result["final_answer"]

        active_session["messages"].append({"role": "assistant", "content": answer})
        active_session["display_messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
