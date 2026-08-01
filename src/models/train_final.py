import joblib

from common import config
from feature_engineering.pipeline import build_pipeline
from preprocessing.data import load_dataset


def train_final(name="tfidf_logreg_balanced"):
    data = load_dataset()
    model = build_pipeline(name)
    model.fit(data["text"], data["label"])
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    return model
