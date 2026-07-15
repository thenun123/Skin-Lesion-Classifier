"""
Shared pytest fixtures for the skin-cancer-api test suite.

pytest automatically discovers this file and makes all fixtures here available
to every test in this package without explicit imports.

sys.path note: backend/src/ is added to sys.path by the `pythonpath = ["src"]`
setting in pyproject.toml, so plain `import api` / `import config` etc. resolve
correctly without any manual sys.path manipulation here.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def app():
    """
    FastAPI application with model loading patched out.

    Session-scoped so the TensorFlow import happens only once across the
    entire test run — avoids 30+ second re-initialisation per test file.
    """
    with patch("model.model_wrapper.load"):
        from api import app as _app
    return _app


@pytest.fixture(scope="session")
def client(app):
    """Synchronous TestClient bound to the patched FastAPI app."""
    return TestClient(app)
