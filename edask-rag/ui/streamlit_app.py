"""Streamlit demo UI for ED(A)SK.

Run with:
    uv run streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from edask.pipeline import ask
from edask.vector_store import get_vector_store

st.set_page_config(page_title="ED(A)SK RAG", page_icon="🔎", layout="wide")

st.title("ED(A)SK")
st.caption("Embeddings → Vector store → Similarity search → Grounded generation")

with st.sidebar:
    st.header("Index status")
    try:
        count = get_vector_store().count()
        st.metric("Stored chunks", count)
    except Exception as exc:
        st.error(f"Could not reach Qdrant: {exc}")
    top_k = st.slider("top-k", min_value=1, max_value=10, value=4)

question = st.text_input("Ask a question", placeholder="e.g. What does an embedding model do?")

if st.button("Ask", type="primary", disabled=not question):
    with st.spinner("Retrieving + generating..."):
        result = ask(question, top_k=top_k)

    st.subheader("Answer")
    st.write(result.answer)

    st.subheader("Citations")
    if not result.citations:
        st.info("No relevant chunks were retrieved.")
    for i, hit in enumerate(result.citations, start=1):
        with st.expander(
            f"[{i}] {hit.source} — chunk {hit.index} — score {hit.score:.3f}"
        ):
            st.write(hit.text)
