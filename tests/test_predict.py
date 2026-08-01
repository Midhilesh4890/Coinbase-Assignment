import pandas as pd
import pytest

from models.predict import load_model, predict, predict_frame


def test_predict_type_errors():
    with pytest.raises(TypeError):
        predict(5)
    with pytest.raises(TypeError):
        predict(None)


def test_predict_value_errors():
    with pytest.raises(ValueError):
        predict("")
    with pytest.raises(ValueError):
        predict("   ")


def test_predict_fraud_message():
    assert predict("someone drained my wallet overnight without authorisation") == "fraud-report"


def test_predict_general_message_label():
    value = predict("how long does identity verification usually take")
    assert value in {"account-access", "fraud-report", "general", "transaction-dispute"}


def test_predict_frame_adds_predictions():
    df = pd.DataFrame({"text": ["hello", "hi there"]})
    result = predict_frame(df, model=make_dummy_model())
    assert "predicted_label" in result.columns
    assert len(result) == len(df)


def test_predict_frame_missing_text_column():
    df = pd.DataFrame({"message": ["hello"]})
    with pytest.raises(KeyError):
        predict_frame(df, model=make_dummy_model())


def test_load_model_returns_predict_method():
    model = load_model()
    assert hasattr(model, "predict")


def make_dummy_model():
    return type("DummyModel", (), {"predict": lambda self, values: ["general"] * len(values)})()
