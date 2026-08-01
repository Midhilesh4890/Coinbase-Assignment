import importlib

from common import config


def test_project_root_override_is_respected(monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", "/tmp/custom-root")
    config_module = importlib.reload(config)
    assert str(config_module.PROJECT_ROOT) == "\\tmp\\custom-root"
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    importlib.reload(config_module)


def test_model_path_ends_with_artifacts_model_joblib():
    assert str(config.MODEL_PATH).endswith("artifacts\\model.joblib")
