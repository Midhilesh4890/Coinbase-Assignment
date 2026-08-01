import joblib

from common import config


def load_model(path=None):
    model_path = config.MODEL_PATH if path is None else path
    return joblib.load(model_path)


def predict(text, model=None):
    if not isinstance(text, str):
        raise TypeError
    if text.strip() == "":
        raise ValueError
    if model is None:
        model = load_model()
    return model.predict([text])[0]
