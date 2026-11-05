import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Tennessee Policy RAG", layout="wide")

st.title("Tennessee Policy RAG")
st.caption("Ask questions about Tennessee State Government policies")

with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Index Document"):
            with st.spinner("Chunking and embedding..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                resp = requests.post(f"{API_URL}/upload", files=files)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Indexed {data['chunks_indexed']} chunks from {data['filename']}")
                else:
                    st.error(f"Error: {resp.text}")
    
    st.divider()
    if st.button("Check API Health"):
        try:
            r = requests.get(f"{API_URL}/health")
            if r.status_code == 200:
                st.success("API is healthy")
            else:
                st.error("API unhealthy")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")

st.header("Ask a Question")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']} — Page {src['page']}**")
                    st.caption(src.get("text", ""))

question = st.chat_input("Ask about Tennessee policies...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "top_k": 5}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    st.markdown(answer)
                    
                    sources = data.get("sources", [])
                    with st.expander("View Sources"):
                        for src in sources:
                            st.markdown(
                                f"**{src['source']} — Page {src['page']}** "
                                f"(score: {src.get('score')})"
                            )
                            st.caption(src.get("text", ""))
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"API error: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")