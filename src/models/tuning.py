import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import Pipeline

from preprocessing.splits import group_folds


def _grid_values():
    return [
        ((1, 1), 0.5, True),
        ((1, 1), 0.5, False),
        ((1, 1), 1, True),
        ((1, 1), 1, False),
        ((1, 1), 2, True),
        ((1, 1), 2, False),
        ((1, 1), 5, True),
        ((1, 1), 5, False),
        ((1, 1), 10, True),
        ((1, 1), 10, False),
        ((1, 2), 0.5, True),
        ((1, 2), 0.5, False),
        ((1, 2), 1, True),
        ((1, 2), 1, False),
        ((1, 2), 2, True),
        ((1, 2), 2, False),
        ((1, 2), 5, True),
        ((1, 2), 5, False),
        ((1, 2), 10, True),
        ((1, 2), 10, False),
        ((1, 3), 0.5, True),
        ((1, 3), 0.5, False),
        ((1, 3), 1, True),
        ((1, 3), 1, False),
        ((1, 3), 2, True),
        ((1, 3), 2, False),
        ((1, 3), 5, True),
        ((1, 3), 5, False),
        ((1, 3), 10, True),
        ((1, 3), 10, False),
    ]


def _build_pipeline(ngram_range, C, sublinear_tf):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(lowercase=True, ngram_range=ngram_range, min_df=1, sublinear_tf=sublinear_tf),
        ),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, C=C, class_weight="balanced")),
    ])


def _score_combination(df, n_splits, seed, ngram_range, C, sublinear_tf):
    folds = group_folds(df, n_splits=n_splits, seed=seed)
    macro_f1_scores = []
    fraud_recall_scores = []
    for train_index, val_index in folds:
        model = _build_pipeline(ngram_range, C, sublinear_tf)
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        model.fit(train_df["text"], train_df["label"])
        pred = model.predict(val_df["text"])
        macro_f1_scores.append(f1_score(val_df["label"], pred, average="macro"))
        fraud_recall_scores.append(
            recall_score(val_df["label"], pred, labels=["fraud-report"], average="macro", zero_division=0)
        )
    return (
        sum(macro_f1_scores) / len(macro_f1_scores),
        pd.Series(macro_f1_scores).std(),
        sum(fraud_recall_scores) / len(fraud_recall_scores),
    )


def grid_search(df, n_splits=5):
    rows = []
    for ngram_range, C, sublinear_tf in _grid_values():
        macro_f1_mean, macro_f1_std, fraud_recall = _score_combination(
            df, n_splits, 42, ngram_range, C, sublinear_tf
        )
        rows.append({
            "ngram_range": ngram_range,
            "C": C,
            "sublinear_tf": sublinear_tf,
            "macro_f1_mean": macro_f1_mean,
            "macro_f1_std": macro_f1_std,
            "fraud_recall": fraud_recall,
        })
    result = pd.DataFrame(rows)
    return (
        result.sort_values(["macro_f1_mean", "macro_f1_std"], ascending=[False, True]).reset_index(drop=True)
    )


def nested_cv(df, n_splits=5, inner_splits=4, inner_seed=1):
    outer_scores = []
    selected = []
    outer_folds = group_folds(df, n_splits=n_splits, seed=42)
    grid = _grid_values()
    for train_index, val_index in outer_folds:
        outer_train = df.iloc[train_index]
        outer_val = df.iloc[val_index]
        best_score = None
        best_combo = None
        for ngram_range, C, sublinear_tf in grid:
            score, _, _ = _score_combination(
                outer_train, inner_splits, inner_seed, ngram_range, C, sublinear_tf
            )
            combo = (ngram_range, C, sublinear_tf)
            if best_score is None or score > best_score:
                best_score = score
                best_combo = combo
        selected.append(best_combo)
        model = _build_pipeline(best_combo[0], best_combo[1], best_combo[2])
        model.fit(outer_train["text"], outer_train["label"])
        pred = model.predict(outer_val["text"])
        outer_scores.append(f1_score(outer_val["label"], pred, average="macro"))
    return {
        "outer_scores": outer_scores,
        "macro_f1_mean": sum(outer_scores) / len(outer_scores),
        "macro_f1_std": pd.Series(outer_scores).std(),
        "selected": selected,
    }
