from __future__ import annotations

"""Retriever implementations **with disk‑level persistence**.

New behaviour
-------------
* **BM25** index is saved to `<corpus>.bm25.pkl` (pickle).
* **FAISS** dense index is saved to `<corpus>.faiss` with a JSON side‑car for
  the corpus size.
* On startup each retriever first looks for the cached file **and only
  rebuilds if it’s missing or the corpus size has changed**.
* All steps log what they are doing so you always know whether we’re *loading*
  or *building*.

Quick example::

    corpus = Corpus("corpus.tsv")
    bm25   = BM25Retriever(corpus)  # loads or builds
    docs   = bm25.retrieve("query", 5)

The code stays dependency‑light and entirely self‑contained; just keep the
TSV and generated cache files together.
"""
from pathlib import Path
from typing import List, Dict, Sequence
import json
import pickle
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

__all__ = [
    "Corpus",
    "BaseRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
]


# --------------------------------------------------------------------------- #
#                                  CORPUS                                     #
# --------------------------------------------------------------------------- #


class Corpus:
    """Thin wrapper around a TSV file with *id* \t *text* \t *title* per line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        print(f"[Corpus] Loading corpus from {self.path}", flush=True)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        # Assume three columns without header
        self.df = pd.read_csv(self.path, sep="\t", names=["id", "text", "title"], dtype=str)
        self.texts: List[str] = self.df["text"].tolist()
        self.ids: List[str] = self.df["id"].tolist()
        self.titles: List[str] = self.df["title"].tolist()
        print(f"[Corpus] Loaded {self.size()} documents", flush=True)

    # ------------------------------ helpers -------------------------------- #
    def size(self) -> int:
        return len(self.df)

    def get_document(self, idx: int) -> Dict[str, str]:
        row = self.df.iloc[idx]
        return {"id": row.id, "text": row.text, "title": row.title}

    def get_documents(self, idxs: Sequence[int]) -> List[Dict[str, str]]:
        return [self.get_document(i) for i in idxs]


# =========================================================================== #
#                                 RETRIEVERS                                  #
# =========================================================================== #


class BaseRetriever:
    """Common interface for all retrievers."""

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        print(f"[{self.__class__.__name__}] Initialising", flush=True)
        self._build_index()

    # ------------------------------------------------------------------- #
    def retrieve(self, query: str, top_k: int = 5):
        print(f"[{self.__class__.__name__}] Retrieving top {top_k} for: '{query}'", flush=True)
        ids = self.retrieve_ids(query, top_k)
        docs = self.corpus.get_documents(ids)
        return docs
        # print(f"Documents retrieved: {docs}", flush=True)
        # texts = [doc["text"] for doc in docs]
        # print(f"Texts retrieved: {texts}", flush=True)

        return texts

    # must override ------------------------------------------------------ #
    def _build_index(self):
        raise NotImplementedError

    def retrieve_ids(self, query: str, top_k: int):
        raise NotImplementedError


# ---------------------------------------------------------------------------- #
#                               BM25 RETRIEVER                                 #
# ---------------------------------------------------------------------------- #


class BM25Retriever(BaseRetriever):
    _bm25_index: BM25Okapi | None = None
    _corpus_size: int | None = None

    def _cache_path(self) -> Path:
        return self.corpus.path.with_suffix(".bm25.pkl")

    def _build_index(self):
        cache = self._cache_path()

        if cache.exists():
            print(f"[BM25Retriever] Found cache at {cache}, loading…", flush=True)
            with cache.open("rb") as f:
                data = pickle.load(f)
            if data["size"] == self.corpus.size():
                BM25Retriever._bm25_index = data["index"]
                BM25Retriever._corpus_size = data["size"]
                print("[BM25Retriever] Cache loaded ✔", flush=True)
            else:
                print("[BM25Retriever] Corpus size changed – rebuilding", flush=True)

        if BM25Retriever._bm25_index is None:
            print("[BM25Retriever] Building BM25 index…", flush=True)
            tokenised = [txt.split() for txt in self.corpus.texts]
            BM25Retriever._bm25_index = BM25Okapi(tokenised)
            BM25Retriever._corpus_size = self.corpus.size()
            # persist --------------------------------------------------
            with cache.open("wb") as f:
                pickle.dump({"size": self.corpus.size(), "index": BM25Retriever._bm25_index}, f)
            print(f"[BM25Retriever] Index built and saved to {cache} ✔", flush=True)
        else:
            print("[BM25Retriever] Using in‑memory BM25 index", flush=True)

        self.bm25: BM25Okapi = BM25Retriever._bm25_index  # type: ignore[assignment]

    def retrieve_ids(self, query: str, top_k: int):
        scores = self.bm25.get_scores(query.split())
        top_idxs = np.argsort(scores)[::-1][:top_k]
        print(f"[BM25Retriever] Indices: {top_idxs.tolist()}", flush=True)
        return top_idxs.tolist()

# ---------------------------------------------------------------------------- #
#                           DENSE (FAISS) RETRIEVER                            #
# ---------------------------------------------------------------------------- #
import re  # NEW – for filename-safe tags
# (rest of your imports stay the same)

class DenseRetriever(BaseRetriever):
    _model: SentenceTransformer | None = None
    _index: faiss.Index | None = None
    _corpus_size: int | None = None
    _model_tag: str | None = None                    # NEW

    # ------------------------ static helpers ----------------------------- #
    @staticmethod
    def _safe_tag(text: str) -> str:                 # NEW
        """Turn 'BAAI/bge-base-en-v1.5' → 'bge_base_en_v1_5' for filenames."""
        return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()

    # --------------------------------------------------------------------- #
    def _cache_path(self) -> Path:
        return self.corpus.path.with_suffix(f".{self._model_tag}.faiss")     # CHANGED

    def _meta_path(self) -> Path:
        return self.corpus.path.with_suffix(f".{self._model_tag}.faiss.json")  # CHANGED

    def _emb_path(self) -> Path:
        return self.corpus.path.with_suffix(f".{self._model_tag}.emb.npy")   # CHANGED

    # --------------------------------------------------------------------- #
    def _build_index(self):
        # ---------------------- model ----------------------------------- #
        if DenseRetriever._model is None:
            print("[DenseRetriever] Loading SentenceTransformer 'BAAI/bge-base-en-v1.5'…", flush=True)
            DenseRetriever._model = SentenceTransformer("BAAI/bge-base-en-v1.5")
            DenseRetriever._model_tag = self._safe_tag("BAAI/bge-base-en-v1.5")  # NEW
            print("[DenseRetriever] Model ready ✔", flush=True)

        self._model_tag = DenseRetriever._model_tag   # NEW

        cache = self._cache_path()
        meta = self._meta_path()
        emb  = self._emb_path()

        cache_ok = cache.exists() and meta.exists()
        if cache_ok:
            with meta.open() as m:
                meta_json = json.load(m)
            if meta_json.get("size") != self.corpus.size():
                cache_ok = False
                print("[DenseRetriever] Corpus size changed – ignoring cached FAISS", flush=True)

        if cache_ok:
            print(f"[DenseRetriever] Loading FAISS index from {cache}…", flush=True)
            DenseRetriever._index = faiss.read_index(str(cache))
            DenseRetriever._corpus_size = meta_json["size"]
            print("[DenseRetriever] FAISS index loaded ✔", flush=True)
        else:
            # ------------------ embeddings ------------------------------ #
            if emb.exists():
                emb_mem = np.load(emb, mmap_mode="r")
                if emb_mem.shape[0] == self.corpus.size():
                    print("[DenseRetriever] Loading cached embeddings …", flush=True)
                    embeddings = emb_mem
                else:
                    print("[DenseRetriever] Cached embeddings size mismatch – recomputing", flush=True)
                    embeddings = None
            else:
                embeddings = None

            if embeddings is None:
                print("[DenseRetriever] Encoding corpus …", flush=True)
                embeddings = DenseRetriever._model.encode(
                    self.corpus.texts,
                    batch_size=64,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                ).astype("float32")
                np.save(emb, embeddings)
                print(f"[DenseRetriever] Embeddings saved to {emb}", flush=True)

            # ------------------ build FAISS index ----------------------- #
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            DenseRetriever._index = index
            DenseRetriever._corpus_size = self.corpus.size()
            # persist -------------------------------------------------- #
            faiss.write_index(index, str(cache))
            with meta.open("w") as m:
                json.dump(
                    {
                        "size": self.corpus.size(),
                        "model": "BAAI/bge-base-en-v1.5",            # NEW (for clarity)
                    },
                    m,
                )
            print(f"[DenseRetriever] Index built and saved to {cache} ✔", flush=True)

        self.model: SentenceTransformer = DenseRetriever._model  # type: ignore[assignment]
        self.index: faiss.Index = DenseRetriever._index          # type: ignore[assignment]

    # ------------------------------------------------------------------- #
    def retrieve_ids(self, query: str, top_k: int):
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        _, idxs = self.index.search(q_emb, top_k)
        print(f"[DenseRetriever] Indices: {idxs[0].tolist()}", flush=True)
        return idxs[0].tolist()
     
# ---------------------------------------------------------------------------- #
#                               HYBRID RETRIEVER                               #
# ---------------------------------------------------------------------------- #

class HybridRetriever(BaseRetriever):
    """Hybrid sparse + dense retrieval with Reciprocal-Rank Fusion.

    score = alpha * RRF_sparse + (1-alpha) * RRF_dense
    """

    def __init__(self, corpus: Corpus, alpha: float = 0.5, rrf_k: int = 60):
        self.alpha = alpha
        self.rrf_k = rrf_k                                   # how steeply RRF decays
        print(f"[HybridRetriever] alpha={alpha}, rrf_k={rrf_k}", flush=True)
        self.bm25  = BM25Retriever(corpus)
        self.dense = DenseRetriever(corpus)
        super().__init__(corpus)

    # No own index -------------------------------------------------------- #
    def _build_index(self):
        print("[HybridRetriever] No additional index needed.", flush=True)

    # ------------------------------ helpers ------------------------------ #
    @staticmethod
    def _rrf(ranks: np.ndarray, k: int):
        """Convert 0-based ranks to RRF scores : 1 / (k + rank)."""
        return 1.0 / (k + ranks)

    # --------------------------------------------------------------------- #
    def retrieve_ids(self, query: str, top_k: int):
        print("[HybridRetriever] Combining sparse & dense (RRF)", flush=True)

        # 1) BM25 scores and ranks --------------------------------------- #
        sparse_scores = self.bm25.bm25.get_scores(query.split())
        sparse_order  = np.argsort(sparse_scores)[::-1]
        ranks_sparse  = np.empty_like(sparse_order)
        ranks_sparse[sparse_order] = np.arange(len(sparse_order))
        rrf_sparse = self._rrf(ranks_sparse, self.rrf_k)

        # 2) Dense scores and ranks -------------------------------------- #
        q_emb = self.dense.model.encode([query], normalize_embeddings=True).astype("float32")
        dense_scores, _ = self.dense.index.search(q_emb, self.corpus.size())
        dense_scores = dense_scores[0]
        dense_order  = np.argsort(dense_scores)[::-1]
        ranks_dense  = np.empty_like(dense_order)
        ranks_dense[dense_order] = np.arange(len(dense_order))
        rrf_dense = self._rrf(ranks_dense, self.rrf_k)

        # 3) Fuse --------------------------------------------------------- #
        combined = self.alpha * rrf_sparse + (1.0 - self.alpha) * rrf_dense
        top_idxs = np.argsort(combined)[::-1][:top_k]

        print(f"[HybridRetriever] Indices: {top_idxs.tolist()}", flush=True)
        return top_idxs.tolist()

def bm25_init():
    """Initialize all retrievers for the NQ dataset."""
    corpus = Corpus("../retriever/passages.tsv")
    bm25 = BM25Retriever(corpus)

    return bm25

def dense_init():
    """Initialize all retrievers for the NQ dataset."""
    corpus = Corpus("../retriever/passages.tsv")
    dense = DenseRetriever(corpus)

    return dense

def hybrid_init():
    """Initialize all retrievers for the NQ dataset."""
    corpus = Corpus("../retriever/passages.tsv")
    hybrid = HybridRetriever(corpus, alpha=0.5)

    return hybrid


# if __name__ == "__main__":
#     corpus = Corpus("passages.tsv")
#     bm25 = BM25Retriever(corpus)
#     dense = DenseRetriever(corpus)
#     hybrid = HybridRetriever(corpus, alpha=0.5)

#     question = "difference between russian blue and british blue cat"
#     top_k = 5
#     print("Using BM25 retriever:")
#     results = bm25.retrieve(question, top_k=top_k)
#     for doc in results:
#         print(doc["id"], doc["title"], doc["text"])

#     print("\nUsing Dense retriever:")
#     results = dense.retrieve(question, top_k=top_k)
#     for doc in results:
#         print(doc["id"], doc["title"], doc["text"])

#     print("\nUsing Hybrid retriever:")
#     results = hybrid.retrieve(question, top_k=top_k)
#     for doc in results:
#         print(doc["id"], doc["title"], doc["text"])



    # corpus   = Corpus("passages.tsv")
    # colbert  = ColBERTRetriever(
    #     corpus,
    #     checkpoint="colbert-ir/colbertv2.0",   # any local or HF checkpoint
    #     root="colbert_runs",                   # where to keep the index
    #     nbits=2                                # product-quantization (2,4,8…)
    # )

    # results = colbert.retrieve(question, top_k=top_k)
    # for doc in results:
    #     print(doc["id"], doc["title"], doc["text"])

    # corpus = Corpus("passages.tsv")
    # bm25 = BM25Retriever(corpus)
    # dense = DenseRetriever(corpus)
    # hybrid = HybridRetriever(corpus, alpha=0.5)

    # question = "difference between russian blue and british blue cat"
    # top_k = 3
    # print("Using BM25 retriever:")
    # results = bm25.retrieve(question, top_k=top_k)
    # # for doc in results:
    # #     print(results)

    # print("\nUsing Dense retriever:")
    # results = dense.retrieve(question, top_k=top_k)
    # # for doc in results:
    # #     print(results)

    # print("\nUsing Hybrid retriever:")
    # results = hybrid.retrieve(question, top_k=top_k)
    # # for doc in results:
    # #     print(results)