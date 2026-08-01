from models.diagnostics import learning_curve, mcnemar, novel_message_check, out_of_fold_predictions
from preprocessing.data import load_dataset


def test_out_of_fold_predictions_length():
    df = load_dataset()
    preds = out_of_fold_predictions(df, "tfidf_logreg_tuned")
    assert len(preds) == len(df)


def test_learning_curve_columns_and_gap():
    curve = learning_curve(load_dataset(), "tfidf_logreg_tuned")
    assert list(curve.columns) == ["fraction", "train_macro_f1", "val_macro_f1", "gap"]
    assert len(curve) == 4
    gaps = list(curve["gap"])
    assert gaps[0] >= gaps[1] >= gaps[2] >= gaps[3]


def test_mcnemar_non_negative_and_range():
    result = mcnemar(load_dataset(), "tfidf_logreg_tuned", "tfidf_logreg_balanced")
    assert isinstance(result["b01"], int)
    assert isinstance(result["b10"], int)
    assert result["b01"] >= 0
    assert result["b10"] >= 0
    assert 0 <= result["p_value"] <= 1


def test_mcnemar_same_pipeline_zero_counts():
    result = mcnemar(load_dataset(), "tfidf_logreg_tuned", "tfidf_logreg_tuned")
    assert result["b01"] == 0
    assert result["b10"] == 0


def test_novel_message_check():
    result = novel_message_check("tfidf_logreg_tuned")
    assert len(result) == 10
    assert result["correct"].all()
