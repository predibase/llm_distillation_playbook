"""Script to generate subsets of the Jigsaw dataset for training and testing for the OpenAI graduation webinar.

Original dataset comes from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

Sample run:
    python jigsaw_data_prep/generate_subsets.py
"""

import pandas as pd
import os


def get_subsets(df, nrows, nsubsets):
    """Get n subsets of the dataframe, df, each of size n."""
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    df_list = []
    for nsubsets in range(nsubsets):
        df_list.append(shuffled_df.iloc[nsubsets * nrows : (nsubsets + 1) * nrows])
    return df_list


def main():
    # Read in raw CSVs.
    train = pd.read_csv("data/original_data/train.csv")
    test = pd.read_csv("data/original_data/test.csv")
    test_labels = pd.read_csv("data/original_data/test_labels.csv")
    # Merge the test and test_labels dataframes.
    test_with_labels = pd.merge(test, test_labels, on="id", how="left")
    # Filter out the rows that have -1, as these have invalid labels (they were intentionally excluded from scoring).
    test_with_labels = test_with_labels[test_with_labels["toxic"] != -1]

    # Create a new column, is_bad, that is True if any of the bad columns are True.
    # All columns that flag that the comment is bad in some way.
    bad_columns_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    train["is_bad"] = train[bad_columns_list].any(axis=1)
    test_with_labels["is_bad"] = test_with_labels[bad_columns_list].any(axis=1)

    # Form pools of data.
    ok_train = train[train["is_bad"] == False]
    bad_train = train[train["is_bad"] == True]
    ok_test = test_with_labels[test_with_labels["is_bad"] == False]
    bad_test = test_with_labels[test_with_labels["is_bad"] == True]

    ok_train_subsets = get_subsets(ok_train, 1000, 3)
    bad_train_subsets = get_subsets(bad_train, 1000, 3)
    ok_test_subsets = get_subsets(ok_test, 200, 3)
    bad_test_subsets = get_subsets(bad_test, 200, 3)

    # Establish a data pool.
    data_pool = {
        # (train) All ok examples.
        "train.a.ok.1k": ok_train_subsets[0],
        "train.b.ok.1k": ok_train_subsets[1],
        # (train) All bad examples.
        "train.a.bad.1k": bad_train_subsets[0],
        "train.b.bad.1k": bad_train_subsets[1],
        "train.c.bad.1k": bad_train_subsets[2],
        # (train) All ok examples.
        "test.a.ok.200": ok_test_subsets[0],
        "test.b.ok.200": ok_test_subsets[1],
        "test.c.ok.200": ok_test_subsets[2],
        # (train) All bad examples.
        "test.a.bad.200": bad_test_subsets[0],
        "test.b.bad.200": bad_test_subsets[1],
        "test.c.bad.200": bad_test_subsets[2],
    }

    # Assemble data subsets.
    dataset_subsets = {
        "sample.tiny": pd.concat([ok_train_subsets[0].sample(frac=0.01), bad_train_subsets[0].sample(frac=0.01)]),
        "test.indist.1": pd.concat([data_pool["test.a.ok.200"], data_pool["test.a.bad.200"].sample(frac=0.1)]),
        "test.indist.2": pd.concat([data_pool["test.b.ok.200"], data_pool["test.b.bad.200"].sample(frac=0.1)]),
        "test.balanced": pd.concat([data_pool["test.c.ok.200"], data_pool["test.c.bad.200"]]),
        "train.5.a": pd.concat([data_pool["train.a.ok.1k"], data_pool["train.a.bad.1k"].sample(frac=0.1)]),
    }
    dataset_subsets["train.5.b"] = pd.concat(
        [dataset_subsets["train.5.a"], data_pool["train.b.ok.1k"], data_pool["train.b.bad.1k"].sample(frac=0.1)]
    )
    dataset_subsets["train.6"] = pd.concat([dataset_subsets["train.5.b"], data_pool["train.c.bad.1k"]])

    # Write out all data.
    os.makedirs("data/data_pools", exist_ok=True)
    os.makedirs("data/dataset_subsets", exist_ok=True)
    for data_pool_id, data_pool_df in data_pool.items():
        data_pool_df.to_csv(f"data/data_pools/{data_pool_id}.csv", index=False)
    for dataset_subset_id, dataset_subset_df in dataset_subsets.items():
        dataset_subset_df.to_csv(f"data/dataset_subsets/{dataset_subset_id}.csv", index=False)


if __name__ == "__main__":
    main()
