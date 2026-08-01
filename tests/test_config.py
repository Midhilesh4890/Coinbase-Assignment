import os
import subprocess
import sys
from pathlib import Path

from common import config


def test_project_root_override_is_respected():
    env = os.environ.copy()
    env["PROJECT_ROOT"] = "/tmp/custom-root"
    result = subprocess.run(
        [sys.executable, "-c", "from common import config; print(config.PROJECT_ROOT)"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == str(Path("/tmp/custom-root"))


def test_model_path_points_at_artifacts_model_joblib():
    assert config.MODEL_PATH.name == "model.joblib"
    assert config.MODEL_PATH.parent.name == "artifacts"
