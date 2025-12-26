# search.py
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="data/chroma",
    embedding_function=embeddings
)

def hybrid_search(query: str, k=5):
    vector_docs = vectordb.similarity_search(query, k=k)

    texts = [d.page_content for d in vector_docs]
    tokenized = [t.split() for t in texts]

    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())

    reranked = sorted(
        zip(vector_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in reranked[:k]]
