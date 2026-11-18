
"""Load and cache the trained hybrid SCD model."""

import streamlit as st
import tensorflow as tf

from src.config import MODEL_PATH


class ModelLoadError(RuntimeError):
    """Raised when the trained model cannot be loaded."""


@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    """Load the saved Keras model once and reuse it."""

    if not MODEL_PATH.is_file():
        raise ModelLoadError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Please check that final_hybrid_model.keras is inside "
            "the model folder."
        )

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        return model

    except Exception as exc:
        raise ModelLoadError(
            f"Unable to load the trained model: {exc}"
        ) from exc
