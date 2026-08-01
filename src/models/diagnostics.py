from scipy.stats import binomtest
import pandas as pd

from feature_engineering.pipeline import build_pipeline
from preprocessing.data import load_dataset
from preprocessing.splits import add_template_key, group_folds


def out_of_fold_predictions(df, name, n_splits=5):
    folds = group_folds(df, n_splits=n_splits)
    pooled = pd.Series(index=df.index, dtype=object)
    for train_index, val_index in folds:
        model = build_pipeline(name)
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        model.fit(train_df["text"], train_df["label"])
        pooled.iloc[val_index] = model.predict(val_df["text"])
    return pooled


def learning_curve(df, name, fractions=(0.25, 0.5, 0.75, 1.0), n_splits=5):
    rows = []
    folds = group_folds(df, n_splits=n_splits)
    for fraction in fractions:
        train_scores = []
        val_scores = []
        for train_index, val_index in folds:
            train_df = add_template_key(df.iloc[train_index])
            val_df = df.iloc[val_index]
            sampled_keys = train_df["template_key"].drop_duplicates().sample(frac=fraction, random_state=0)
            sampled_df = train_df[train_df["template_key"].isin(sampled_keys)]
            if sampled_df["label"].nunique() < 4:
                continue
            model = build_pipeline(name)
            model.fit(sampled_df["text"], sampled_df["label"])
            train_pred = model.predict(sampled_df["text"])
            val_pred = model.predict(val_df["text"])
            train_scores.append(pd.Series(sampled_df["label"]).eq(train_pred).mean())
            val_scores.append(pd.Series(val_df["label"]).eq(val_pred).mean())
        train_macro_f1 = sum(train_scores) / len(train_scores) if train_scores else 0
        val_macro_f1 = sum(val_scores) / len(val_scores) if val_scores else 0
        rows.append({"fraction": fraction, "train_macro_f1": train_macro_f1, "val_macro_f1": val_macro_f1, "gap": train_macro_f1 - val_macro_f1})
    return pd.DataFrame(rows)


def mcnemar(df, name_a, name_b, n_splits=5):
    pred_a = out_of_fold_predictions(df, name_a, n_splits=n_splits)
    pred_b = out_of_fold_predictions(df, name_b, n_splits=n_splits)
    correct_a = pred_a.eq(df["label"])
    correct_b = pred_b.eq(df["label"])
    b01 = int((~correct_a & correct_b).sum())
    b10 = int((correct_a & ~correct_b).sum())
    return {
        "name_a": name_a,
        "name_b": name_b,
        "errors_a": int((~correct_a).sum()),
        "errors_b": int((~correct_b).sum()),
        "b01": b01,
        "b10": b10,
        "p_value": binomtest(b01, b01 + b10, 0.5).pvalue if (b01 + b10) else 1.0,
    }


def novel_message_check(name, model=None):
    messages = [
        ("someone drained my wallet overnight without authorisation", "fraud-report"),
        ("unauthorised party moved my coins to an address I do not recognise", "fraud-report"),
        ("i think a criminal took over my portfolio and stole everything", "fraud-report"),
        ("two factor code never arrives so i am locked out", "account-access"),
        ("cannot sign in, the verification step keeps failing", "account-access"),
        ("i sent money three days ago and it never arrived, want it back", "transaction-dispute"),
        ("charged twice for the same purchase, need a refund", "transaction-dispute"),
        ("what are the fees for converting between currencies", "general"),
        ("do you support customers living in norway", "general"),
        ("how long does identity verification usually take", "general"),
    ]
    if model is None:
        data = load_dataset()
        model = build_pipeline(name)
        model.fit(data["text"], data["label"])
    texts = [item[0] for item in messages]
    expected = [item[1] for item in messages]
    predicted = model.predict(texts)
    return pd.DataFrame({"text": texts, "expected": expected, "predicted": predicted, "correct": [a == b for a, b in zip(expected, predicted)]})
