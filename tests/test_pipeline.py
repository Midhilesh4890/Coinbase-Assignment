from feature_engineering.pipeline import PIPELINES, build_pipeline


def test_pipeline_names():
    assert set(PIPELINES) == {"tfidf_logreg", "tfidf_logreg_balanced", "tfidf_logreg_tuned"}


def test_build_pipeline_returns_fit_and_predict():
    for name in PIPELINES:
        pipeline = build_pipeline(name)
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict")


def test_build_pipeline_unknown_name_raises():
    import pytest

    with pytest.raises(KeyError):
        build_pipeline("unknown")


def test_tuned_pipeline_parameters():
    pipeline = build_pipeline("tfidf_logreg_tuned")
    params = pipeline.get_params()
    assert params["tfidf__ngram_range"] == (1, 1)
    assert params["tfidf__sublinear_tf"] is False
    assert params["logreg__C"] == 2
    assert params["logreg__class_weight"] == "balanced"
