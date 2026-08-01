from pathlib import Path

import pandas as pd

from models.evaluate import confusion, cross_validate, log_experiment
from preprocessing.data import load_dataset


def test_cross_validate_shape_and_range():
    result = cross_validate(load_dataset(), "tfidf_logreg_tuned")
    assert len(result["per_fold"]) == 5
    assert 0 <= result["macro_f1_mean"] <= 1


def test_tuned_pipeline_high_score():
    result = cross_validate(load_dataset(), "tfidf_logreg_tuned")
    assert result["macro_f1_mean"] >= 0.95


def test_confusion_square_and_labels():
    df = confusion(load_dataset(), "tfidf_logreg_tuned")
    assert df.shape[0] == df.shape[1]
    assert list(df.index) == ["account-access", "fraud-report", "general", "transaction-dispute"]
    assert list(df.columns) == ["account-access", "fraud-report", "general", "transaction-dispute"]


def test_log_experiment_writes_tmp_csv_and_not_reports(tmp_path, monkeypatch):
    repo_reports = Path(r"D:\Projects\Coinbase-Assignment\reports\experiments.csv")
    before = repo_reports.read_bytes() if repo_reports.exists() else None
    monkeypatch.chdir(tmp_path)
    result = cross_validate(load_dataset(), "tfidf_logreg_tuned")
    log_experiment(result)
    path = Path("reports/experiments.csv")
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == 1
    if before is None:
        assert not repo_reports.exists()
    else:
        assert repo_reports.read_bytes() == before
