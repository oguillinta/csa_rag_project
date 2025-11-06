import os
import pandas as pd
import streamlit as st

from config import CHROMA_DIR, COLLECTION_NAME, SIMILARITY_THRESHOLD, TOP_K
from data_loader import csv_to_documents
from vector_store import get_collection, clear_collection, add_batch, query_by_embedding
from rag_service import get_client, embed_texts, best_hits, answer_with_rag

st.set_page_config(page_title="Customer Support Agent Chatbot", page_icon="🛎️", layout="wide")
st.title("Customer Support Agent (RAG)")
st.caption("CSV → Chroma → OpenAI. Answers only within support scope.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY",""))
    n_results = st.slider("Top-K documents", 1, 10, TOP_K)
    threshold = st.slider("Similarity threshold", 0.20, 0.95, SIMILARITY_THRESHOLD, step=0.01)
    persist_dir = st.text_input("Chroma folder", value=CHROMA_DIR)

    st.markdown("---")
    uploaded = st.file_uploader("Upload a CSV (orders/inquiries)", type=["csv"], accept_multiple_files=True)
    rebuild = st.button("🔁 Rebuild Index (from uploads or sample)")
    clear_chat = st.button("🧹 Clear chat")

# prepare vector collection
collection = get_collection(persist_dir, COLLECTION_NAME)

def rebuild_index():
    collection = clear_collection(persist_dir, COLLECTION_NAME)

    docs, metas, ids = [], [], []

    if not uploaded:
        st.info("No uploads provided. Using sample: data/orders.csv")
        df = pd.read_csv("data/orders.csv")
        d, m, i = csv_to_documents(df, "orders.csv")
        docs.extend(d); metas.extend(m); ids.extend(i)
    else:
        for f in uploaded:
            df = pd.read_csv(f)
            d, m, i = csv_to_documents(df, f.name)
            docs.extend(d); metas.extend(m); ids.extend(i)

    client = get_client(api_key)
    embs = embed_texts(client, docs)
    add_batch(collection, docs, metas, ids, embs)
    return len(docs)

if rebuild:
    with st.spinner("Building index..."):
        count = rebuild_index()
    st.success(f"Indexed {count} rows.")

# chat state
if clear_chat or "msgs" not in st.session_state:
    st.session_state.msgs = []

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask about orders, shipping, cancellations, returns...")
if question:
    st.session_state.msgs.append({"role":"user","content":question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            client = get_client(api_key)
            q_text = question.lower()
            q_emb = embed_texts(client, [q_text])[0]
            results = query_by_embedding(collection, q_emb, n_results=n_results)
            hits = best_hits(results)
            if hits:
                with st.expander("Retrieved context (top-K)"):
                    for doc, meta, sim in hits:
                        st.write(f"{round(sim,3)} • {meta.get('source','unknown')} (row {meta.get('row_index')})")
            answer = answer_with_rag(client, question, hits, threshold)
            st.markdown(answer)
            st.session_state.msgs.append({"role":"assistant","content":answer})
        except Exception as e:
            st.error(str(e))
