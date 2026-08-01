import joblib

from common import config


def load_model(path=None):
    model_path = config.MODEL_PATH if path is None else path
    return joblib.load(model_path)


def predict(text, model=None):
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    if text.strip() == "":
        raise ValueError("text must not be empty or whitespace-only")
    if model is None:
        model = load_model()
    return model.predict([text])[0]


def predict_frame(df, model=None):
    if "text" not in df.columns:
        raise KeyError('dataframe must contain a "text" column')
    if model is None:
        model = load_model()
    result = df.copy()
    result["predicted_label"] = model.predict(result["text"])
    return result
