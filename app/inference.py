
"""Run single-image predictions using the trained hybrid model."""

import numpy as np

from src.config import INPUT_SHAPE, THRESHOLD


class PredictionError(RuntimeError):
    """Raised when prediction cannot be completed."""


class Prediction:
    def __init__(self, probability: float, threshold: float):
        self.probability = probability
        self.label = (
            "Sickle class"
            if probability >= threshold
            else "Normal class"
        )


def predict(model, rgb_image: np.ndarray) -> Prediction:
    """Predict the class probability for one preprocessed RGB image."""

    if not isinstance(rgb_image, np.ndarray):
        raise PredictionError("Invalid image data.")

    if rgb_image.shape != INPUT_SHAPE:
        raise PredictionError(
            f"Expected image shape {INPUT_SHAPE}, "
            f"but received {rgb_image.shape}."
        )

    try:
        image = rgb_image.astype(np.float32)
        batch = np.expand_dims(image, axis=0)

        output = model(batch, training=False)
        probability = float(np.asarray(output).reshape(-1)[0])

        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise PredictionError(
                "The model returned an invalid probability."
            )

        return Prediction(probability, THRESHOLD)

    except PredictionError:
        raise
    except Exception as exc:
        raise PredictionError(
            f"Prediction failed: {exc}"
        ) from exc
