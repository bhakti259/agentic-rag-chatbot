import streamlit as st
import uuid
from backend.rag_graph import app as rag_app
from backend.btw_handler import handle_off_topic

st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="📄")
st.title("📄 Agentic RAG Chatbot")

# Initialize session state — only runs once per browser session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Ask a question about your paper...")

if user_query:
    # Show user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Check off-topic first
    off_topic_response = handle_off_topic(user_query)

    if off_topic_response:
        answer = off_topic_response
    else:
        # Run through the main graph
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

    # Show assistant's response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)