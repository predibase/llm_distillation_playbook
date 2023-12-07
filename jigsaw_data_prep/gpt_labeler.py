"""Sample run:

python jigsaw_data_prep/gpt_labeler.py \
    --outdir=data/dataset_subsets/ \
    --llm=gpt-3.5-turbo \
    --input_file=data/dataset_subsets/sample.tiny.csv
    
python jigsaw_data_prep/gpt_labeler.py \
    --outdir=data/dataset_subsets/ \
    --llm=gpt-3.5-turbo \
    --input_file=data/dataset_subsets/test.balanced.csv
"""

import argparse
import csv
import logging
import os
from alive_progress import alive_bar
import dotenv
import openai
from openai_multi_client import OpenAIMultiClient
from tabulate import tabulate
from eval_gpt_label_quality import calculate_accuracy_and_roc
import pandas as pd

# Environment variables.
dotenv.load_dotenv()
logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPEN_AI_KEY")

PROMPTS = {
    "simple_prompt": """Given the input text below, please classify whether it is toxic. Provide your classification in a JSON format with the key 'is_bad'. If the comment is indeed toxic, set is_bad=1, otherwise set is_bad=0.

Input text: '%s'

Output:""",
    #     "cot_prompt": """Given the input text below, please determine if it is toxic. Please format your response in the form of a python dictionary with the following keys:
    # 1. 'reason_innocent': Provide a reason why the comment might be innocent. For parsing consistency, do not use any single or double quotes in your response string, even if the your text may be grammatically invalid.
    # 2. 'reason_toxic': Provide a reason why the comment might be toxic. For parsing consistency, do not use single or double quotes in your response string, even if the your text may be grammatically invalid.
    # 3. 'is_bad': Your final judgment on whether the comment is toxic or not. If the comment is indeed toxic, use set is_bad=1, otherwise set is_bad=0.
    # Input text: '%s'
    # Output:""",
    #     "cot_prompt": """For a well-intentioned content moderation app, we want to flag a comment if it may contain toxic, insulting, obscene, or threatening language. Given the input text, please classify its toxicity. Please format your response in the form of a python dictionary with the following keys:
    # 1. 'is_bad': If the comment is indeed toxic, use set is_bad=1, otherwise set is_bad=0.
    # 2. 'reason': Provide an appropriate amount of detail for for your judgment.
    # Input text: '%s'
    # Output:""",
}


def generate_synthetic_data(args):
    """Generate synthetic data for a given task."""
    api = OpenAIMultiClient(
        endpoint="chats",
        data_template={"model": args.llm},
        max_retries=3,
        retry_max=10,
    )

    num_failed_queries = 0

    # Parse the input file
    data_file_dict = {}
    num_rows = 0
    row_id = 0
    with open(args.input_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Make this configurable.
            data_file_dict[row_id] = row
            num_rows += 1
            row_id += 1

    def make_requests():
        for row_id, values in data_file_dict.items():
            for prompt_id, prompt in PROMPTS.items():
                full_prompt = prompt % values["comment_text"]

                api.request(
                    data={
                        "messages": [
                            {
                                "role": "user",
                                "content": full_prompt,
                            }
                        ]
                    },
                    metadata={
                        "row_id": row_id,
                        "prompt_id": prompt_id,
                    },
                )

    api.run_request_function(make_requests)

    # Parse results out.
    with alive_bar(num_rows * len(list(PROMPTS.keys()))) as progress_bar:
        for result in api:
            try:
                response = result.response["choices"][0]["message"]["content"]
                row_id = result.metadata["row_id"]
                prompt_id = result.metadata["prompt_id"]
                data_file_dict[row_id][prompt_id] = response
            except Exception as e:
                logger.warning(f"Failed to parse response {response}, which failed with error: {e}")
                num_failed_queries += 1
            progress_bar()

    # Determine the output directory.
    outfile_basename = f"{os.path.basename(args.input_file).replace('.csv', '')}.{args.llm}"
    outdir_basename = os.path.join(args.outdir, f"{os.path.basename(args.input_file).replace('.csv', '')}.{args.llm}")
    counter = 0
    if os.path.exists(outdir_basename):
        outdir_basename = f"{outdir_basename}_{counter}"
        while os.path.exists(outdir_basename):
            outdir_basename = outdir_basename.replace(f"_{counter}", f"_{counter + 1}")
            counter += 1

    # Write out the prompts.
    os.makedirs(os.path.join(outdir_basename, "prompts"), exist_ok=True)
    for prompt_id, prompt in PROMPTS.items():
        with open(os.path.join(outdir_basename, f"prompts/{prompt_id}.txt"), "w") as f:
            f.write(prompt)

    with open(
        os.path.join(outdir_basename, f"{outfile_basename}.with_labels.csv"),
        "w",
    ) as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=[*data_file_dict[0].keys()],
        )
        csv_writer.writeheader()

        for row_id, values in data_file_dict.items():
            csv_writer.writerow(values)

    # Calculate metrics and write them to a separate file.
    data = pd.read_csv(os.path.join(outdir_basename, f"{outfile_basename}.with_labels.csv"))
    prompt_ids = PROMPTS.keys()
    table_data = []
    headers = ["prompt_id", "accuracy", "fpr", "tpr", "roc_auc"]
    for prompt_id in prompt_ids:
        accuracy, fpr, tpr, roc_auc = calculate_accuracy_and_roc(data, prompt_id)
        table_data.append([prompt_id, accuracy, fpr, tpr, roc_auc])

    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    print(f"Total num_failed_queries: {num_failed_queries}")
    with open(os.path.join(outdir_basename, "metrics.txt"), "w") as f:
        f.write(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
        f.write("\n")
        f.write(f"Total num_failed_queries: {num_failed_queries}")


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    generate_synthetic_data(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Multi-threaded GPT Labeler",
        description="Get labels from GPT.",
    )
    parser.add_argument("--outdir", help="Output file.", required=True)
    parser.add_argument(
        "--llm",
        default="gpt-3.5-turbo",
        help="LLM to use.",
        required=True,
    )
    parser.add_argument(
        "--input_file",
        help="Input file.",
        required=True,
    )

    args = parser.parse_args()
    main(args)
