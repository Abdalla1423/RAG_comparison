from enum import Enum

class MODELS(Enum):
    GPT_4 = 'GPT_4'
    LLAMA_8B = 'LLAMA_8B'
    LLAMA_70B = 'LLAMA_70B'


model_map = {
    "gpt4": MODELS.GPT_4.value,
    "llama8b": MODELS.LLAMA_8B.value,
    "llama70b": MODELS.LLAMA_70B.value,  # Assuming llama70b is also treated as LLAMA_8B for simplicity
}


class RETRIEVER(Enum):
    BM25 = 'BM25'
    DENSE = 'DENSE'
    HYBRID = 'HYBRID'


retriever_map = {
    "bm25": RETRIEVER.BM25.value,
    "dense": RETRIEVER.DENSE.value,
    "hybrid": RETRIEVER.HYBRID.value,
}
