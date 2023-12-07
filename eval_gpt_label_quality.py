"""Sample usage:

python eval_gpt_label_quality.py \
    --input_file=data/dataset_subsets/sample.tiny.with_synthetic_labels.csv
"""

import argparse
import ast
from sklearn.metrics import roc_curve, auc
import pandas as pd
from tabulate import tabulate


def calculate_accuracy_and_roc(df, prompt_output_column: str):
    # Function to safely parse string as a Python dictionary and extract 'is_bad'
    def safe_eval(s):
        try:
            return ast.literal_eval(s)["is_bad"]
        except (ValueError, SyntaxError):
            print(f"Encountered ValueError when parsing: {s}")
            return None  # or False, depending on how you want to handle invalid strings

    # Parse strings safely and extract 'is_bad'
    parsed_is_bad_column_name = f"{prompt_output_column}.parsed.is_bad"
    df[parsed_is_bad_column_name] = df[prompt_output_column].apply(safe_eval)

    # Handle rows with invalid strings (if any)
    # For example, here we drop rows with None in 'predicted_is_bad'
    # df = df.dropna(subset=[parsed_is_bad_column_name])

    # Convert boolean 'is_bad' column to integer for comparison
    df["is_bad"] = df["is_bad"].astype(bool)
    df[parsed_is_bad_column_name] = df[parsed_is_bad_column_name].astype(bool)

    # Calculate accuracy
    accuracy = (df[parsed_is_bad_column_name] == df["is_bad"]).mean()

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(df["is_bad"], df[parsed_is_bad_column_name])
    roc_auc = auc(fpr, tpr)

    return accuracy, fpr, tpr, roc_auc


def main(args):
    data = pd.read_csv(args.input_file)

    prompt_ids = ["simple_prompt", "cot_prompt"]

    table_data = []
    headers = ["prompt_id", "accuracy", "fpr", "tpr", "roc_auc"]

    print(f"Metrics for: '{args.input_file}':")
    for prompt_id in prompt_ids:
        accuracy, fpr, tpr, roc_auc = calculate_accuracy_and_roc(data, prompt_id)
        table_data.append([prompt_id, accuracy, fpr, tpr, roc_auc])

    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Evaluate GPT label quality.",
        description="Get labels from GPT.",
    )
    parser.add_argument(
        "--input_file",
        help="Input file.",
        required=True,
    )

    args = parser.parse_args()
    main(args)
