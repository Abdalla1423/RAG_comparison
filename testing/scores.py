import os
import argparse
import pandas as pd
from datasets import Dataset
from ragas.metrics import answer_correctness
from ragas import evaluate

from testing.type_definitions import model_map, retriever_map
from dotenv import load_dotenv

load_dotenv()


# ---------- small helper ----------------------------------------------------
def compute_answer_correctness(question: str, answer: str, gold: str) -> float:
    """
    Wraps RAGAS evaluation for a single (q, a, gold) triple.
    Returns the scalar answer_correctness score (0-1).
    """
    ds = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "ground_truth": [gold],
        }
    )
    result = evaluate(ds, metrics=[answer_correctness])
    # evaluate() returns a dict like {'answer_correctness': 0.9876}
    return float(result["answer_correctness"][0])


# ---------- main loop -------------------------------------------------------
def calculate_answer_correctness(model: str, retrievers: list[str]) -> None:
    """
    For every strategy (prompt framework) associated with the chosen model:
      • open / create the results sheet
      • compute RAGAS answer_correctness for every unprocessed row
      • write the file back after each row
    """
    for retriever in retrievers:
        file_path = f"clapnq/{model}_{retriever}_CLAPNQ.xlsx"

        # (Re)open the sheet or create an empty one
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
        else:
            df = pd.DataFrame(
                columns=["Id", "Question", "Retrieved Informatio", "doc-id-list", "Gold doc-id-list", "Answer", "Gold Answer", "Answer Correctness"]
            )

        # make sure the target column exists
        if "Answer Correctness" not in df.columns:
            df["Answer Correctness"] = pd.NA

        # iterate row-by-row
        for idx, row in df.iterrows():
            # 1) skip what is already done
            if not pd.isna(row["Answer Correctness"]) and row["Answer Correctness"] != -1:
                continue

            q, a, gold = row["Question"], row["Answer"], row["Gold Answer"]

            try:
                score = compute_answer_correctness(q, a, gold)
            except Exception as exc:
                print(f"[ERROR] Row {idx}: {exc}")
                score = -1

            # 2) store the metric
            df.loc[idx, "Answer Correctness"] = score

            # 3) save immediately
            df.to_excel(file_path, index=False)
            print(
                f"Iteration {idx + 1}: saved answer_correctness={score:.4f} "
                f"for question '{q[:60]}…'"
            )

        print(
            f"All questions processed for strategy '{retriever}' with model '{model}'."
        )

def calculate_average_answer_correctness(model: str, retrievers: list[str]) -> None:
    """
    For every strategy (prompt framework) associated with the chosen model:
      • open / create the results sheet
      • compute RAGAS answer_correctness for every unprocessed row
      • write the file back after each row
    """
    for retriever in retrievers:
        file_path = f"clapnq/{model}_{retriever}_CLAPNQ.xlsx"

        # (Re)open the sheet or create an empty one
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
        else:
            df = pd.DataFrame(
                columns=["Id", "Question", "Retrieved Information", "doc-id-list", "Gold doc-id-list", "Answer", "Gold Answer", "Answer Correctness"]
            )

        # make sure the target column exists
        if "Answer Correctness" not in df.columns:
            df["Answer Correctness"] = pd.NA

        # calculate average correctness
        avg_correctness = df["Answer Correctness"].mean()
        print(f"Average answer correctness for {model} with {retriever}: {avg_correctness:.4f}")

# ---------- CLI wrapper (all original flags preserved) ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["llama8b", "llama70b"],
        required=True,
        help="Which model to use? Example:  --model gpt4",
    )
    parser.add_argument(
        "--retriever",
        nargs="+",
        choices=["serper","bm25", "dense", "hybrid"],
        required=True,
        help="One or more retrieval strategies to evaluate.",
    )
    args = parser.parse_args()

    # initialise model exactly as before
    model = args.model

    # translate strategy names → prompt framework ids
    retriever = [retriever_map[s] for s in args.retriever]

    # run the new metric
    calculate_answer_correctness(model_map[model], retriever)
    calculate_average_answer_correctness(model_map[model], retriever)


if __name__ == "__main__":
    main()
