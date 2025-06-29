from testing.type_definitions import model_map, retriever_map
import pandas as pd
import time
import os
import argparse
from retriever.retriever import set_retriever, retrieve
from openai import OpenAI
import textwrap


NUM_OF_STATEMENTS = 300

#   This function uses the specified prompt framework (pf) to determine the veracity
#   of a given claim. If a 'name' is provided, it prepends that name to the claim text
#   before sending it to the framework. It then attempts to parse the result as JSON
#   to extract a 'label' and an 'explanation'.


def build_prompt(context: list[str], question: str) -> list[dict]:
    joined = "\n\n---------\n\n".join(context)
    return [
        {"role": "system", "content": "You are a concise, factual QA assistant. Use only the provided context to answer the question."},
        {"role": "user",   "content": textwrap.dedent(f"""
            Context:
            {joined}

            Question: {question}
            Answer in 2-4 sentences.
        """)}
    ]


def askLlamaKISS(question: str, context: list[str]):
    openai_api_key = "ace0b273a98ae4ab26103f97a775bdf1"
    openai_api_base = "https://chat-ai.academiccloud.de/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    completion = client.chat.completions.create(
        model="meta-llama-3.1-8b-instruct",
        messages=build_prompt(context, question),
        temperature=0,
    )

    return completion.choices[0].message.content


def preprocess_clapnq():
    sampled_data_file = 'data/question_dev_answerable.xlsx'
    sampled_data = pd.read_excel(sampled_data_file)
    sampled_data = sampled_data.head(NUM_OF_STATEMENTS)
    return sampled_data


#   Runs a specific evidence retrieval / prompt framework strategy against the AVERITEC
#   dataset. It processes each statement in the dataset, skipping those already completed
#   in an existing output file. The results are continuously written to an Excel file.


def evaluate_strategy(model, retriever):
    sampled_data = preprocess_clapnq()
    output_file_path = f'{model}_{retriever}_CLAPNQ.xlsx'

    if os.path.exists(output_file_path):
        evaluated_data = pd.read_excel(output_file_path)
    else:
        evaluated_data = pd.DataFrame(columns=[
                                      'Id', 'Question', 'Retrieved Information', 'doc-id-list', 'Gold doc-id-list', 'Answer', 'Gold Answer'])

    start_time = time.time()

    for index, row in sampled_data.iterrows():
        # Check if the current statement has already been processed (to avoid duplicates)
        id = row['id']
        question = row['question']
        gold_answer = row['answers']
        gold_doc_id_list = row['doc-id-list']

        filtered_data = evaluated_data[evaluated_data['Question'] == row['question']]
        if not filtered_data.empty:
            print(f"Statement {row['question']} already processed. Skipping.")
            continue

        retrieved_information, doc_ids = retrieve(question)
        generated_answer = askLlamaKISS(question, retrieved_information)

        temp_df = pd.DataFrame({
            'Id': [id],
            'Question': question,
            'Retrieved Information': [retrieved_information],
            'doc-id-list': [doc_ids],
            'Gold doc-id-list': [gold_doc_id_list],
            'Answer': [generated_answer],
            'Retrieved Information': [retrieved_information[:]],
            'Gold Answer': [gold_answer]
        })

        # If the statement existed (with incorrect form), replace its row, else append

        evaluated_data = pd.concat(
            [evaluated_data, temp_df], ignore_index=True)
        print(
            f'Iteration {index+1}: Results appended and saved for question "{question}"')

        evaluated_data.to_excel(output_file_path, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(
        f'All statements processed for strategy "{retriever}" and dataset "AVERITEC". Elapsed time: {elapsed_time} s')
    return evaluated_data

#   The main entry point for running the full pipeline:
#   1. Parses CLI arguments to choose a model and one or more strategies.
#   2. Sets the selected model and retriever globally.
#   3. Calls 'evaluate_strategy' for each chosen strategy.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["gpt4", "llama8b", "llama70b"],
        required=True,
        help="Which model to use? Options: 'gpt4', 'llama8b', 'llama70b' . Example usage: python main.py --model gpt4"
    )

    parser.add_argument(
        "--retriever",
        choices=["serper", "bm25", "dense", "hybrid"],
        required=True,
        help="Which retriever to use? Options: 'serper', 'bm25', 'dense', 'hybrid'. Example usage: python main.py --retriever serper"
    )
    args = parser.parse_args()

    chosen_model = model_map[args.model]

    chosen_retriever = retriever_map[args.retriever]
    set_retriever(chosen_retriever)

    evaluate_strategy(chosen_model, chosen_retriever)


if __name__ == "__main__":
    main()
