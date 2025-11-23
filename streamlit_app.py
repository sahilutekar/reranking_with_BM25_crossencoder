import streamlit as st
import nltk
nltk.download("punkt", quiet=True)

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ------------------------------------------
# DOCUMENTS (same as your FastAPI code)
# ------------------------------------------
docs = [
    # 0
    "Transformers are deep learning architectures based entirely on attention mechanisms. "
    "They removed the need for recurrent networks by allowing models to process all tokens simultaneously. "
    "Transformers now dominate NLP tasks such as translation, summarization, and language modeling.",

    # 1
    "Recurrent Neural Networks (RNNs) and their variants like LSTMs and GRUs were traditionally used for "
    "sequence modeling tasks because they processed input tokens step-by-step. However, they struggled with "
    "long-range dependencies and parallelization.",

    # 2
    "Support Vector Machines (SVMs) are classical machine learning models that separate data points using "
    "maximum-margin hyperplanes. They are not used for sequence modeling in modern NLP systems.",

    # 3
    "Convolutional Neural Networks (CNNs) are widely used in vision tasks and sometimes for text classification. "
    "However, CNNs did not replace RNNs in tasks requiring long-range context understanding.",

    # 4
    "The Transformer architecture introduced multi-head attention, which enables capturing global contextual "
    "information more effectively than recurrence. This led to its adoption as the successor to RNN-based models.",

    # 5
    "BM25 is a classical information retrieval algorithm that ranks documents based on bag-of-words scoring. "
    "It is not related to sequence modeling or neural architectures.",

    # 6
    "GRUs are a simplified form of LSTMs that help overcome vanishing gradients in recurrent training. "
    "Although they improved training stability, they still rely on token-by-token processing.",

    # 7
    "Transformer-based large language models such as BERT, GPT-2, and GPT-4 demonstrate the superiority of "
    "attention-based processing over recurrent architectures across nearly all NLP tasks.",

    # 8
    "Neural networks are general function approximators inspired by the human brain. They include many architectures "
    "such as MLPs, CNNs, RNNs, and Transformers.",

    # 9
    "LSTMs used gating mechanisms to capture relatively long-term dependencies, but still could not compete with the "
    "parallel computation abilities of Transformer models.",

    # 10
    "Hybrid architectures attempted to combine RNNs with attention but eventually gave way to pure Transformer models "
    "because of efficiency and scalability.",

    # 11
    "Modern NLP pipelines rely heavily on Transformer encoders and decoders. These have replaced recurrent models "
    "largely due to improved generalization and training parallelization."
]


# ------------------------------------------
# BM25 SETUP
# ------------------------------------------
tokenized = [d.split() for d in docs]
bm25 = BM25Okapi(tokenized)

def bm25_get_candidates(query: str, k: int = 5):
    q_tok = query.split()
    scores = bm25.get_scores(q_tok)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(idx, float(scores[idx])) for idx in ranked]


# ------------------------------------------
# CROSS-ENCODER SETUP
# ------------------------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, idxs, top_n=3):
    pairs = [(query, docs[i]) for i in idxs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(idxs, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [(i, float(s)) for i, s in ranked]


# ------------------------------------------
# STREAMLIT UI
# ------------------------------------------
st.set_page_config(page_title="Semantic Search (BM25 + CrossEncoder)", layout="wide")

st.title("🔍 Semantic Search Engine")
st.subheader("BM25 Retrieval + CrossEncoder Semantic Reranking")

query = st.text_input("Enter your query")

k = st.slider("Number of BM25 candidates", 1, 10, 5)
top_n = st.slider("Top results after Reranking", 1, 5, 3)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        st.write("### 🟦 Step 1: BM25 Candidate Retrieval")
        bm25_result = bm25_get_candidates(query, k)
        for idx, score in bm25_result:
            st.write(f"**Doc {idx} — Score: {score:.4f}**\n\n{docs[idx]}\n\n---")

        st.write("### 🟩 Step 2: CrossEncoder Semantic Reranking")
        idxs = [i for i, _ in bm25_result]
        reranked = rerank(query, idxs, top_n)

        for idx, score in reranked:
            st.write(f"**Doc {idx} — Semantic Score: {score:.4f}**\n\n{docs[idx]}\n\n---")

        # Best match
        best_idx, best_score = reranked[0]
        st.success("### 🟧 Best Match")
        st.write(f"**Document {best_idx}** — Score: `{best_score:.4f}`")
        st.write(docs[best_idx])
