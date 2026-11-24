from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLE_LOGS = Path(__file__).resolve().parents[1] / "app" / "samples" / "sample_logs.txt"


@pytest.fixture(autouse=True)
def isolated_model_path(tmp_path, monkeypatch):
    """Every test gets its own model file so tests never touch models/."""
    path = tmp_path / "model.joblib"
    monkeypatch.setenv("LOG_ANALYZER_MODEL_PATH", str(path))
    return path


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_logs():
    return SAMPLE_LOGS.read_text()
