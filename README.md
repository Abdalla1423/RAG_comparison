# RAG Benchmark (BEIR‑NQ ➜ Llama)

End‑to‑end grid evaluator for Retrieval‑Augmented Generation.
Evaluates every (retriever, model) pair you enable – or a single pair –
using **BEIR Natural‑Questions test split** and **RAGAS answer_correctness**.

* 2 × LLMs : Llama 8B, 70B (behind an OpenAI‑compatible endpoint)
* 3 × Retrievers : sparse (BM25), dense (FAISS), hybrid (score fusion)

```bash
# quickstart – create env, install, run one combo
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# build indexes once and run dense+70B
python -m src.run retriever=dense model=llama-70b
