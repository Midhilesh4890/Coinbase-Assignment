import pandas as pd

from common import config
from models import train_final as train_final_module
from statistical_analysis.statistics import class_distribution, duplication_stats, label_conflicts


def test_class_distribution():
    df = load_df()
    result = class_distribution(df)
    assert set(result.columns) == {"label", "count", "proportion"}
    assert result["count"].sum() == len(df)


def test_duplication_stats():
    df = load_df()
    result = duplication_stats(df)
    assert result["n_rows"] == len(df)
    assert result["n_unique_texts"] <= len(df)
    assert result["n_unique_keys"] <= len(df)
    assert 0 <= result["duplication_rate"] <= 1


def test_label_conflicts():
    df = pd.DataFrame(
        {
            "text": ["a", "a"],
            "label": ["general", "fraud-report"],
        }
    )
    result = label_conflicts(df)
    assert len(result) == 1
    assert result.loc[0, "row_count"] == 2
    assert result.loc[0, "labels"] == ["fraud-report", "general"]


def test_train_final_uses_artifact_paths(monkeypatch, tmp_path):
    dumped = {}

    def fake_dump(model, path):
        dumped["path"] = path
        return path

    monkeypatch.setattr(train_final_module.joblib, "dump", fake_dump)
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "artifacts" / "model.joblib")
    model = train_final_module.train_final()
    assert hasattr(model, "predict")
    assert dumped["path"] == config.MODEL_PATH


def load_df():
    return pd.DataFrame(
        {
            "text": [
                "hello",
                "hello",
                "need help with withdrawal",
                "need help with withdrawal",
            ],
            "label": ["general", "general", "account-access", "fraud-report"],
        }
    )
