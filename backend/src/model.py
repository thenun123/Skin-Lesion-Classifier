"""
Model loading and lifecycle management.

Deliberately separate from inference.py: this module owns *loading and
validating* the model artefact; inference.py owns *using* it. That separation
keeps training/experimentation code and serving code decoupled.
"""
import logging
import os
import threading
from typing import Optional

import tensorflow as tf

from config import config

logger = logging.getLogger("skin_cancer_api")


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction request arrives before the model has loaded."""


class ModelWrapper:
    """
    Thread-safe singleton that loads the Keras model exactly once per process.

    The singleton pattern prevents the ~5 s TensorFlow cold-start from blocking
    multiple concurrent startup attempts, and keeps GPU/memory usage bounded.
    """

    _instance: Optional["ModelWrapper"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ModelWrapper":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._model = None
                cls._instance._loaded = False
        return cls._instance

    def load(self) -> None:
        """
        Load and validate the model from disk.

        Validates input shape and output dimensionality at load time so that a
        mismatched model file fails loudly at startup rather than silently
        producing garbage predictions at inference time.
        """
        if self._loaded:
            return

        if not os.path.exists(config.MODEL_PATH):
            raise ModelNotLoadedError(
                f"Model file not found at '{config.MODEL_PATH}'. "
                f"Copy your trained '{config.MODEL_FILENAME}' into the models/ directory. "
                f"See README.md for instructions."
            )

        logger.info("Loading model from %s ...", config.MODEL_PATH)
        self._model = tf.keras.models.load_model(config.MODEL_PATH)

        # Validate input shape — catches wrong model files before they cause
        # silent prediction errors in production.
        expected_hw = config.IMG_SIZE
        actual_shape = self._model.input_shape
        if tuple(actual_shape[1:3]) != tuple(expected_hw):
            logger.warning(
                "Loaded model expects input shape %s, but config.IMG_SIZE is %s. "
                "Predictions may be invalid — verify you copied the correct model file.",
                actual_shape,
                expected_hw,
            )

        # Validate output shape — must be a single sigmoid unit for binary classification.
        output_shape = self._model.output_shape
        if output_shape[-1] != 1:
            logger.warning(
                "Expected a single sigmoid output unit (binary classifier), "
                "but the loaded model's output shape is %s.",
                output_shape,
            )

        self._loaded = True
        logger.info("Model loaded successfully.")

    @property
    def model(self) -> tf.keras.Model:
        """Return the loaded Keras model. Raises ModelNotLoadedError if not yet loaded."""
        if not self._loaded or self._model is None:
            raise ModelNotLoadedError(
                "Model is not loaded. Ensure the API startup event ran and the "
                "model file exists in the models/ directory."
            )
        return self._model

    def is_ready(self) -> bool:
        """Return True if the model has been loaded and validated successfully."""
        return self._loaded


model_wrapper = ModelWrapper()
