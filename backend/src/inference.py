"""
Inference pipeline: raw image bytes → preprocessed tensor → prediction dict.

CRITICAL — Preprocessing parity contract:
    This function must stay in lockstep with `decode_and_resize()` used in the
    training and error-analysis notebooks:

        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)   # stays in [0, 255]

    No manual /255 rescale is applied here — the Custom CNN has its own internal
    `Rescaling(1./255)` layer as its first layer. Feeding pre-scaled [0, 1] input
    would rescale twice and silently produce garbage predictions.
"""
import time
from typing import Any

import tensorflow as tf

from config import config
from model import model_wrapper


def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    """
    Decode arbitrary JPEG/PNG bytes into the tensor format the model expects.

    Output shape: (1, 224, 224, 3), dtype float32, values in [0, 255].
    Matches the training-time preprocessing pipeline exactly.
    """
    img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    # bilinear resize — matches the tf.image.resize default used during training
    img = tf.image.resize(img, config.IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)  # (H, W, C) → (1, H, W, C)
    return img


def predict(image_bytes: bytes) -> dict[str, Any]:
    """
    Run a single end-to-end prediction and return a JSON-serialisable result dict.

    Args:
        image_bytes: Raw bytes of a JPEG or PNG image.

    Returns:
        A dict with keys: label, probability_malignant, confidence,
        threshold, inference_time_ms.
    """
    start = time.perf_counter()

    tensor = preprocess_image(image_bytes)
    prob_malignant: float = float(
        model_wrapper.model.predict(tensor, verbose=0).ravel()[0]
    )

    label_idx: int = int(prob_malignant >= config.THRESHOLD)
    label: str = config.CLASS_NAMES[label_idx]
    confidence: float = prob_malignant if label_idx == 1 else (1.0 - prob_malignant)

    elapsed_ms: float = (time.perf_counter() - start) * 1000

    return {
        "label": label,
        "probability_malignant": round(prob_malignant, 4),
        "confidence": round(confidence, 4),
        "threshold": config.THRESHOLD,
        "inference_time_ms": round(elapsed_ms, 2),
    }
