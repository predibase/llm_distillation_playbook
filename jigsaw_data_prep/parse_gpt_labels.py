""""Script to parse GPT labels from the 'simple_prompt' column of the dataset."""

import pandas as pd
import ast
import re
import sys


def find_is_bad_value(s):
    # Use a regular expression to find 'is_bad' (in single or double quotes) and its numeric value
    match = re.search(r"('is_bad'|\"is_bad\"):\s*(\d+)", s)

    # Check if the pattern was found
    if match:
        # Return the numeric value as an integer
        return int(match.group(2))
    else:
        # Return None or raise an error if 'is_bad' is not found
        return None


def parse_gpt_labels_as_separate_column(df, prompt_output_column: str, parsed_is_bad_column_name: str):
    # Function to safely parse string as a Python dictionary and extract 'is_bad'
    def safe_eval(s):
        try:
            return ast.literal_eval(s)["is_bad"]
        except (ValueError, SyntaxError):
            # print(f"Encountered ValueError when parsing: {s}")
            try:
                is_bad = find_is_bad_value(s)
                return is_bad
            except:
                print(f"Encountered error when parsing: {s}, and could not manually extract value.")
                return None

    # Parse strings safely and extract 'is_bad'
    df[parsed_is_bad_column_name] = df[prompt_output_column].apply(safe_eval)

    df["is_bad"] = df["is_bad"].astype(bool)
    df[parsed_is_bad_column_name] = df[parsed_is_bad_column_name].astype(bool)

    return df


def main(argv):
    dataset_paths = [
        "data/dataset_subsets/train.5.a.gpt-4-1106-preview/train.5.a.gpt-4-1106-preview.with_labels.csv",
        "data/dataset_subsets/train.5.b.gpt-4-1106-preview/train.5.b.gpt-4-1106-preview.with_labels.csv",
        "data/dataset_subsets/train.6.gpt-4-1106-preview/train.6.gpt-4-1106-preview.with_labels.csv",
        "data/dataset_subsets/train.5k.gpt-4-1106-preview/train.5k.gpt-4-1106-preview.with_labels.csv",
        "data/dataset_subsets/train.10k.gpt-4-1106-preview/train.10k.gpt-4-1106-preview.with_labels.csv",
        "data/dataset_subsets/train.5.c.csv",
    ]

    for dataset_path in dataset_paths:
        data = pd.read_csv(dataset_path)
        data = parse_gpt_labels_as_separate_column(data, "simple_prompt", "is_bad_gpt")
        data.to_csv(dataset_path.replace(".csv", ".parsed.csv"))


if __name__ == "__main__":
    main(sys.argv)
