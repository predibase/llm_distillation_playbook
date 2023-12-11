import pandas as pd
import ast
import re


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


def keep_correct_generated_responses(df, prompt_output_column: str):
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
    parsed_is_bad_column_name = f"{prompt_output_column}.parsed.is_bad"
    df[parsed_is_bad_column_name] = df[prompt_output_column].apply(safe_eval)

    # Convert boolean 'is_bad' column to integer for comparison
    df["is_bad"] = df["is_bad"].astype(bool)
    df[parsed_is_bad_column_name] = df[parsed_is_bad_column_name].astype(bool)

    # Filter out rows where the generated value is different from the ground truth.
    df = df[df[parsed_is_bad_column_name] == df["is_bad"]]
    return df


data = pd.read_csv("data/dataset_subsets/train.5.b.gpt-4-1106-preview/train.5.b.gpt-4-1106-preview.with_labels.csv")
original_len = len(data)

data = keep_correct_generated_responses(data, "simple_prompt")

data.to_csv("data/dataset_subsets/train.5.c.csv")
new_len = len(data)

print(f"Removed {original_len - new_len} rows.")

original_data = pd.read_csv("data/original_data/train.csv")

random_additional_data = original_data.sample(n=original_len - new_len)

equalized_data = pd.concat([data, random_additional_data])
equalized_data.to_csv("data/dataset_subsets/train.5.d.csv")
