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
    """Thin wrapper around a TSV file with *doc_id* \t *text* \t *title* per line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        print(f"[Corpus] Loading corpus from {self.path}", flush=True)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        # Assume three columns without header
        self.df = pd.read_csv(self.path, sep="\t", names=["doc_id", "text", "title"], dtype=str)
        self.texts: List[str] = self.df["text"].tolist()
        self.doc_ids: List[str] = self.df["doc_id"].tolist()
        self.titles: List[str] = self.df["title"].tolist()
        print(f"[Corpus] Loaded {self.size()} documents", flush=True)

    # ------------------------------ helpers -------------------------------- #
    def size(self) -> int:
        return len(self.df)

    def get_document(self, idx: int) -> Dict[str, str]:
        row = self.df.iloc[idx]
        return {"doc_id": row.doc_id, "text": row.text, "title": row.title}

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
        return self.corpus.get_documents(ids)

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


class DenseRetriever(BaseRetriever):
    _model: SentenceTransformer | None = None
    _index: faiss.Index | None = None
    _corpus_size: int | None = None

    def _cache_path(self) -> Path:
        return self.corpus.path.with_suffix(".faiss")

    def _meta_path(self) -> Path:
        return self.corpus.path.with_suffix(".faiss.json")

    def _build_index(self):
        # ---------------------- model ----------------------------------- #
        if DenseRetriever._model is None:
            print("[DenseRetriever] Loading SentenceTransformer 'all-MiniLM-L6-v2'…", flush=True)
            DenseRetriever._model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[DenseRetriever] Model ready ✔", flush=True)

        cache = self._cache_path()
        meta = self._meta_path()
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
            # ------------------ build fresh ----------------------------- #
            print(f"[DenseRetriever] Building FAISS index for {self.corpus.size()} docs…", flush=True)
            embeddings = DenseRetriever._model.encode(
                self.corpus.texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            ).astype("float32")
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            DenseRetriever._index = index
            DenseRetriever._corpus_size = self.corpus.size()
            # persist -------------------------------------------------- #
            faiss.write_index(index, str(cache))
            with meta.open("w") as m:
                json.dump({"size": self.corpus.size()}, m)
            print(f"[DenseRetriever] Index built and saved to {cache} ✔", flush=True)

        self.model: SentenceTransformer = DenseRetriever._model  # type: ignore[assignment]
        self.index: faiss.Index = DenseRetriever._index  # type: ignore[assignment]

    def retrieve_ids(self, query: str, top_k: int):
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        _, idxs = self.index.search(q_emb, top_k)
        print(f"[DenseRetriever] Indices: {idxs[0].tolist()}", flush=True)
        return idxs[0].tolist()


# ---------------------------------------------------------------------------- #
#                               HYBRID RETRIEVER                               #
# ---------------------------------------------------------------------------- #


class HybridRetriever(BaseRetriever):
    """Linear fusion of sparse + dense scores (alpha·sparse + (1‑alpha)·dense)."""

    def __init__(self, corpus: Corpus, alpha: float = 0.5):
        self.alpha = alpha
        print(f"[HybridRetriever] alpha={alpha}", flush=True)
        self.bm25 = BM25Retriever(corpus)
        self.dense = DenseRetriever(corpus)
        super().__init__(corpus)

    def _build_index(self):
        print("[HybridRetriever] No additional index", flush=True)

    # helpers ------------------------------------------------------------- #
    @staticmethod
    def _min_max(arr: np.ndarray):
        min_v, max_v = arr.min(), arr.max()
        return (arr - min_v) / (max_v - min_v) if max_v != min_v else np.zeros_like(arr)

    def retrieve_ids(self, query: str, top_k: int):
        print("[HybridRetriever] Combining sparse & dense", flush=True)
        sparse = self.bm25.bm25.get_scores(query.split())
        q_emb = self.dense.model.encode([query], normalize_embeddings=True).astype("float32")
        dense, _ = self.dense.index.search(q_emb, self.corpus.size())
        dense = dense[0]
        combined = self.alpha * self._min_max(sparse) + (1 - self.alpha) * self._min_max(dense)
        top_idxs = np.argsort(combined)[::-1][:top_k]
        print(f"[HybridRetriever] Indices: {top_idxs.tolist()}", flush=True)
        return top_idxs.tolist()



corpus = Corpus("passages.tsv")
bm25 = BM25Retriever(corpus)
dense = DenseRetriever(corpus)
hybrid = HybridRetriever(corpus, alpha=0.5)

question = "difference between russian blue and british blue cat"
top_k = 10
print("Using BM25 retriever:")
results = bm25.retrieve(question, top_k=top_k)
for doc in results:
    print(doc["doc_id"], doc["title"], doc["text"])

print("\nUsing Dense retriever:")
results = dense.retrieve(question, top_k=top_k)
for doc in results:
    print(doc["doc_id"], doc["title"], doc["text"])

print("\nUsing Hybrid retriever:")
results = hybrid.retrieve(question, top_k=top_k)
for doc in results:
    print(doc["doc_id"], doc["title"], doc["text"])