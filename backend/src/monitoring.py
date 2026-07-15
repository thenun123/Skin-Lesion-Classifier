"""
Lightweight monitoring: structured request logging to a file, plus an
in-memory rolling window that powers the /metrics summary endpoint.

Deliberately dependency-free (no Prometheus client, no external services) so
this runs anywhere the rest of the API runs. If you later wire into
Prometheus/Grafana/Datadog, this is the single module to swap.
"""
import json
import logging
import os
import time
from collections import deque
from threading import Lock
from typing import Any

from config import config

os.makedirs(config.LOG_DIR, exist_ok=True)

logger = logging.getLogger("skin_cancer_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _fmt = logging.Formatter("%(asctime)s %(message)s")

    _file_handler = logging.FileHandler(config.LOG_FILE)
    _file_handler.setFormatter(_fmt)
    logger.addHandler(_file_handler)

    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(_fmt)
    logger.addHandler(_stream_handler)


class Monitor:
    """
    Thread-safe rolling-window metrics collector.

    Keeps the last `window_size` prediction results in memory and exposes
    aggregate statistics via the /metrics endpoint. All counter increments
    are guarded by a lock for safe concurrent access.
    """

    def __init__(self, window_size: int = config.METRICS_WINDOW_SIZE) -> None:
        self._lock: Lock = Lock()
        self._window: deque[dict[str, Any]] = deque(maxlen=window_size)
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._start_time: float = time.time()

    def record_prediction(self, result: dict[str, Any], filename: str | None = None) -> None:
        """Record a successful prediction into the rolling window and write a structured log."""
        with self._lock:
            self._total_requests += 1
            self._window.append(result)
        logger.info(json.dumps({"event": "prediction", "filename": filename, **result}))

    def record_error(self, error_message: str, context: str = "") -> None:
        """Increment the error counter and write a structured error log."""
        with self._lock:
            self._total_errors += 1
        logger.error(json.dumps({"event": "error", "context": context, "message": error_message}))

    def summary(self) -> dict[str, Any]:
        """Return aggregate metrics over the current rolling window."""
        with self._lock:
            window = list(self._window)
            total = self._total_requests
            errors = self._total_errors

        base: dict[str, Any] = {
            "total_requests": total,
            "total_errors": errors,
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "recent_predictions_tracked": len(window),
            "avg_inference_time_ms": None,
            "recent_malignant_rate": None,
        }

        if window:
            avg_latency = sum(r["inference_time_ms"] for r in window) / len(window)
            malignant_count = sum(1 for r in window if r["label"] == "malignant")
            base["avg_inference_time_ms"] = round(avg_latency, 2)
            base["recent_malignant_rate"] = round(malignant_count / len(window), 3)

        return base


monitor = Monitor()
