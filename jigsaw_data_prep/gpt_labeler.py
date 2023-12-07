"""Sample run command:
python jigsaw_data_prep/gpt_labeler.py --outdir=data/dataset_subsets/ --llm=gpt-3.5-turbo --input_file=data/dataset_subsets/sample.tiny.csv
"""

import argparse
import csv
import logging
import os
from alive_progress import alive_bar
import dotenv
import openai
from openai_multi_client import OpenAIMultiClient

# Environment variables.
dotenv.load_dotenv()
logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPEN_AI_KEY")


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
            content = f"""For a well-intentioned content moderation use case, given the input text below, please classify whether it is toxic. Provide your classification in a JSON format with keys 'is_bad' and 'reason'. If the comment is indeed toxic, use set is_bad=1, otherwise set is_bad=0. For the 'reason', please include an appropriate amount of detail for how you determined the toxicity of the input text.

Input text: '{values["comment_text"]}'

Output:"""
            api.request(
                data={
                    "messages": [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ]
                },
                metadata={
                    "row_id": row_id,
                },
            )

    api.run_request_function(make_requests)

    with open(os.path.join(args.outdir, "synthetic_data.csv"), "w") as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=["generated_output"] + list(data_file_dict[0].keys()),
        )
        csv_writer.writeheader()
        with alive_bar(num_rows) as progress_bar:
            for result in api:
                try:
                    response = result.response["choices"][0]["message"]["content"]
                    row_id = result.metadata["row_id"]
                    csv_writer.writerow({"generated_output": response, **data_file_dict[row_id]})
                except Exception as e:
                    logger.warning(f"Failed to parse response {response}, which failed with error: {e}")
                    num_failed_queries += 1
                progress_bar()

    print(f"Total num_failed_queries: {num_failed_queries}")


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
