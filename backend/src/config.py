"""
Central configuration for the Skin Cancer Classification API.

All values are overridable via environment variables so the same code
runs unchanged across local dev, Docker, and any future deployment target.

Directory layout (resolved at runtime):
    backend/
    ├── src/config.py   ← this file
    ├── models/         ← MODEL_DIR default
    └── logs/           ← LOG_DIR default
"""
import os
from pathlib import Path

# parent       = backend/src/
# parent.parent = backend/
BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    # --- Model ---
    # Matches the winning model from the training notebook:
    # Custom CNN (highest F1 = 0.8118 on the held-out test set).
    MODEL_DIR: str = os.getenv("MODEL_DIR", str(BASE_DIR / "models"))
    MODEL_FILENAME: str = os.getenv("MODEL_FILENAME", "Custom_CNN.keras")
    MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_FILENAME)

    # --- Preprocessing (must match training notebooks exactly) ---
    IMG_SIZE: tuple[int, int] = (224, 224)
    CLASS_NAMES: list[str] = ["benign", "malignant"]  # index 0 = benign, index 1 = malignant
    POSITIVE_CLASS: str = "malignant"

    # Threshold tuned to 0.34 via precision-recall analysis on the validation set.
    # A missed malignant carries higher clinical cost than a false alarm —
    # shifting below 0.5 improves recall on the positive class.
    THRESHOLD: float = float(os.getenv("PREDICTION_THRESHOLD", "0.34"))

    # --- Upload validation ---
    ALLOWED_CONTENT_TYPES: set[str] = {"image/jpeg", "image/png", "image/jpg"}
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))

    # --- Logging / monitoring ---
    LOG_DIR: str = os.getenv("LOG_DIR", str(BASE_DIR / "logs"))
    LOG_FILE: str = os.path.join(LOG_DIR, "predictions.log")
    METRICS_WINDOW_SIZE: int = int(os.getenv("METRICS_WINDOW_SIZE", "500"))

    # --- Server ---
    APP_ENV: str = os.getenv("APP_ENV", "dev")  # "dev" | "production"
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # --- API metadata ---
    API_TITLE: str = "Skin Cancer Classification API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = (
        "Binary skin lesion classifier (benign vs malignant) served from a "
        "Custom CNN trained on an ISIC-derived dermoscopy dataset. "
        "Research/demo use only — not a medical device."
    )
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")


config = Config()
