
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def _tfidf_logreg():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("logreg", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def _tfidf_logreg_balanced():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("logreg", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ])


def _tfidf_logreg_tuned():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 1), min_df=1, sublinear_tf=False)),
        ("logreg", LogisticRegression(max_iter=1000, random_state=42, C=2, class_weight="balanced")),
    ])


PIPELINES = {
    "tfidf_logreg": _tfidf_logreg,
    "tfidf_logreg_balanced": _tfidf_logreg_balanced,
    "tfidf_logreg_tuned": _tfidf_logreg_tuned,
}


def build_pipeline(name):
    return PIPELINES[name]()

