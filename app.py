# app.py
import streamlit as st
from llm_factory import get_llm
from search import hybrid_search
from ingestion import ingest_pdf

st.set_page_config(page_title="PDF Chatbot", layout="wide")

st.title("📄 PDF AI Chatbot")

# Sidebar
with st.sidebar:
    llm_name = st.selectbox(
        "Choose LLM",
        ["typhoon", "qwen", "gpt"]
    )

    uploaded = st.file_uploader("Upload PDF", type="pdf")
    if uploaded:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded.read())
        ingest_pdf("temp.pdf")
        st.success("PDF ingested")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

query = st.chat_input("Ask something from your document")

if query:
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    docs = hybrid_search(query)
    context = "\n\n".join([d.page_content for d in docs])

    llm = get_llm(llm_name)

    prompt = f"""
    Answer ONLY from the context.
    If not found, say you don't know.

    Context:
    {context}

    Question:
    {query}
    """

    answer = llm.invoke(prompt).content

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    st.chat_message("assistant").write(answer)
