import pandas as pd

from preprocessing.dedup import template_key


def class_distribution(df):
    counts = df.groupby("label").size().reset_index(name="count")
    counts["proportion"] = counts["count"] / len(df)
    return counts


def duplication_stats(df):
    keys = df["text"].map(template_key)
    return {
        "n_rows": len(df),
        "n_unique_texts": df["text"].nunique(),
        "n_unique_keys": keys.nunique(),
        "duplication_rate": 1 - keys.nunique() / len(df),
        "max_rows_per_key": keys.value_counts().max(),
    }


def label_conflicts(df):
    keys = df["text"].map(template_key)
    temp = df.assign(_key=keys)
    grouped = temp.groupby("_key")
    rows = []
    for key, group in grouped:
        labels = sorted(group["label"].unique())
        if len(labels) > 1:
            rows.append({"template_key": key, "row_count": len(group), "labels": labels})
    return pd.DataFrame(rows)
