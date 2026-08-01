from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

from feature_engineering.pipeline import build_pipeline
from preprocessing.splits import group_folds


def cross_validate(df, name, n_splits=5):
    folds = group_folds(df, n_splits=n_splits)
    pooled = pd.Series(index=df.index, dtype=object)
    per_fold = []
    macro_f1_scores = []
    fraud_recall_scores = []
    for fold, (train_index, val_index) in enumerate(folds):
        model = build_pipeline(name)
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        model.fit(train_df["text"], train_df["label"])
        pred = model.predict(val_df["text"])
        pooled.iloc[val_index] = pred
        macro_f1 = f1_score(val_df["label"], pred, average="macro")
        fraud_recall = recall_score(
            val_df["label"], pred, labels=["fraud-report"], average="macro", zero_division=0
        )
        macro_f1_scores.append(macro_f1)
        fraud_recall_scores.append(fraud_recall)
        per_fold.append({"fold": fold, "macro_f1": macro_f1, "fraud_recall": fraud_recall})
    report = classification_report(df["label"], pooled)
    return {
        "name": name,
        "macro_f1_mean": sum(macro_f1_scores) / len(macro_f1_scores),
        "macro_f1_std": pd.Series(macro_f1_scores).std(),
        "fraud_recall_mean": sum(fraud_recall_scores) / len(fraud_recall_scores),
        "fraud_recall_std": pd.Series(fraud_recall_scores).std(),
        "per_fold": per_fold,
        "report": report,
    }


def confusion(df, name, n_splits=5):
    folds = group_folds(df, n_splits=n_splits)
    pooled = pd.Series(index=df.index, dtype=object)
    for train_index, val_index in folds:
        model = build_pipeline(name)
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        model.fit(train_df["text"], train_df["label"])
        pooled.iloc[val_index] = model.predict(val_df["text"])
    labels = sorted(df["label"].unique())
    matrix = confusion_matrix(df["label"], pooled, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def log_experiment(result):
    path = Path("reports/experiments.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "timestamp": datetime.now(UTC).isoformat(),
        "name": result["name"],
        "macro_f1_mean": result["macro_f1_mean"],
        "macro_f1_std": result["macro_f1_std"],
        "fraud_recall_mean": result["fraud_recall_mean"],
        "fraud_recall_std": result["fraud_recall_std"],
        "params": str(build_pipeline(result["name"]).get_params()),
    }])
    if path.exists():
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, index=False)
