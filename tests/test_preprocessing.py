from preprocessing.data import load_dataset, load_holdout
from preprocessing.dedup import template_key
from preprocessing.splits import add_template_key, group_folds


def test_load_dataset_shape_and_columns():
    df = load_dataset()
    assert len(df) == 400
    assert list(df.columns) == ["text", "label"]


def test_expected_labels_present():
    df = load_dataset()
    assert set(df["label"]) == {"account-access", "fraud-report", "general", "transaction-dispute"}


def test_template_key_collapses_examples():
    a = "Hello team, My withdrawal of 2 ETH shows completed but I never received it. Thanks."
    b = "Hey, My withdrawal of 5 BTC shows completed but I never received it. Appreciate any help."
    assert template_key(a) == template_key(b)


def test_template_key_idempotent():
    value = template_key(
        "Hello team, My withdrawal of 2 ETH shows completed but I never received it. Thanks."
    )
    assert template_key(value) == value.lower()


def test_add_template_key_adds_column():
    df = load_dataset().head(5)
    result = add_template_key(df)
    assert "template_key" in result.columns
    assert len(result) == len(df)


def test_group_folds_properties():
    df = add_template_key(load_dataset())
    folds = group_folds(df)
    assert len(folds) == 5
    seen = set()
    labels = {"account-access", "fraud-report", "general", "transaction-dispute"}
    for train_index, val_index in folds:
        assert set(df.iloc[train_index]["template_key"]).isdisjoint(set(df.iloc[val_index]["template_key"]))
        assert set(df.iloc[val_index]["label"]) == labels
        for index in val_index:
            assert index not in seen
            seen.add(index)
    assert len(seen) == len(df)


def test_load_holdout_returns_text_column():
    df = load_holdout("data/sample_holdout.csv")
    assert "text" in df.columns
