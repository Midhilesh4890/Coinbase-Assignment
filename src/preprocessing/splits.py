from sklearn.model_selection import StratifiedGroupKFold

from preprocessing.dedup import template_key


def add_template_key(df):
    result = df.copy()
    result["template_key"] = result["text"].map(template_key)
    return result


def group_folds(df, n_splits=5, seed=42):
    if "template_key" not in df.columns:
        df = add_template_key(df)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(df, df["label"], df["template_key"]))
