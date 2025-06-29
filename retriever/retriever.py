from retriever.nq_retriever import bm25_init, dense_init, hybrid_init

current_retriever = "BM25"  # Default retriever
top_k = 5  # Default number of top results to retrieve
bm25_retriever = None
dense_retriever = None
hybrid_retriever = None

def set_retriever(retriever):
    global current_retriever, bm25_retriever, dense_retriever, hybrid_retriever
    current_retriever = retriever
    if current_retriever == "BM25":
        bm25_retriever = bm25_init()
    elif current_retriever == "DENSE":
        dense_retriever = dense_init()
    elif current_retriever == "HYBRID":
        hybrid_retriever = hybrid_init()

def retrieve(query):
    if current_retriever == "BM25":
        result = bm25_retriever.retrieve(query, top_k=top_k)
    elif current_retriever == "DENSE":
        result = dense_retriever.retrieve(query, top_k=top_k)
    elif current_retriever == "HYBRID":
        result = hybrid_retriever.retrieve(query, top_k=top_k)
    else:
        raise ValueError(f"Unknown retriever: {current_retriever}. Please set a valid retriever.")
    texts = [doc["text"] for doc in result]
    doc_ids = [doc["id"] for doc in result]
    return texts, doc_ids

