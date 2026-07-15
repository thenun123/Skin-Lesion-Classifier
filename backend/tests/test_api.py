"""
API endpoint test suite.

Imports work because pyproject.toml sets pythonpath = ["src"], which adds
backend/src/ to sys.path automatically when pytest is invoked from backend/.

All tests mock model loading at import time so the lifespan handler never
attempts to load a real .keras file during CI or local test runs.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Patch model loading before the app module is imported.
# The session-scoped `client` fixture in conftest.py is the canonical way to
# get the test client — used by all tests below.
with patch("model.model_wrapper.load"):
    from api import app as _app

_client = TestClient(_app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_model_not_loaded() -> None:
    """Health endpoint returns model_not_loaded when wrapper is not ready."""
    with patch("model.model_wrapper.is_ready", return_value=False):
        response = _client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "model_not_loaded"


def test_health_model_loaded() -> None:
    """Health endpoint returns ok when wrapper is ready."""
    with patch("model.model_wrapper.is_ready", return_value=True):
        response = _client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── /metrics ──────────────────────────────────────────────────────────────────

def test_metrics_initial_state() -> None:
    """Metrics endpoint returns all required fields on a fresh process."""
    response = _client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "total_errors" in data
    assert "uptime_seconds" in data


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_model_not_loaded() -> None:
    """Predict endpoint returns 503 when the model has not loaded."""
    with patch("model.model_wrapper.is_ready", return_value=False):
        response = _client.post(
            "/predict",
            files={"file": ("test.jpg", b"fake image content", "image/jpeg")},
        )
    assert response.status_code == 503
    assert "Model is not loaded" in response.json()["detail"]


def test_predict_invalid_file_type() -> None:
    """Predict endpoint returns 400 for unsupported MIME types."""
    with patch("model.model_wrapper.is_ready", return_value=True):
        response = _client.post(
            "/predict",
            files={"file": ("test.txt", b"some text", "text/plain")},
        )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_predict_file_too_large() -> None:
    """Predict endpoint returns 413 when the upload exceeds MAX_UPLOAD_SIZE_MB."""
    with patch("model.model_wrapper.is_ready", return_value=True):
        large_content = b"0" * (11 * 1024 * 1024)  # 11 MB > 10 MB limit
        response = _client.post(
            "/predict",
            files={"file": ("large.jpg", large_content, "image/jpeg")},
        )
    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]


def test_predict_success() -> None:
    """Predict endpoint returns the correct payload on a valid upload."""
    mock_prediction = {
        "label": "benign",
        "probability_malignant": 0.1234,
        "confidence": 0.8766,
        "threshold": 0.34,
        "inference_time_ms": 12.34,
    }
    with (
        patch("model.model_wrapper.is_ready", return_value=True),
        patch("api.predict", return_value=mock_prediction),
    ):
        response = _client.post(
            "/predict",
            files={"file": ("skin_lesion.jpg", b"fake jpeg data", "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "benign"
    assert data["probability_malignant"] == 0.1234
    assert data["confidence"] == 0.8766
    assert data["threshold"] == 0.34
    assert data["inference_time_ms"] == 12.34
