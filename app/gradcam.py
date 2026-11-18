
"""Grad-CAM-style visualization for the EfficientNet branch."""

import cv2
import numpy as np
import tensorflow as tf

from src.config import EFFNET_LAYER, GRADCAM_LAYER, IMG_SIZE


def compute_gradcam(model, rgb_image: np.ndarray) -> np.ndarray:
    """
    Generate a normalized activation map from the EfficientNet branch.

    This follows the notebook implementation: gradients are calculated
    from the mean EfficientNet branch output, rather than the final
    hybrid classification score.
    """

    if not isinstance(rgb_image, np.ndarray):
        raise ValueError("Invalid image data.")

    if rgb_image.shape != (IMG_SIZE, IMG_SIZE, 3):
        raise ValueError(
            f"Expected image shape {(IMG_SIZE, IMG_SIZE, 3)}."
        )

    try:
        efficientnet = model.get_layer(EFFNET_LAYER)
        last_conv = efficientnet.get_layer(GRADCAM_LAYER)

        grad_model = tf.keras.Model(
            inputs=efficientnet.input,
            outputs=[last_conv.output, efficientnet.output]
        )

        image = rgb_image[None].astype(np.float32)

        with tf.GradientTape() as tape:
            conv_output, branch_output = grad_model(
                image,
                training=False
            )
            loss = tf.reduce_mean(branch_output)

        gradients = tape.gradient(loss, conv_output)

        if gradients is None:
            raise ValueError(
                "Could not calculate gradients for the activation map."
            )

        weights = tf.reduce_mean(
            gradients,
            axis=(1, 2)
        )

        cam = tf.reduce_sum(
            conv_output * weights[:, None, None, :],
            axis=-1
        )[0]

        cam = tf.maximum(cam, 0)
        cam = cam / (tf.reduce_max(cam) + 1e-8)

        cam = cv2.resize(
            cam.numpy(),
            (IMG_SIZE, IMG_SIZE)
        )

        return np.clip(cam, 0, 1).astype(np.float32)

    except Exception as exc:
        raise RuntimeError(
            f"Could not generate the activation map: {exc}"
        ) from exc


def overlay(rgb_image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend a JET-coloured activation map with the original RGB image."""

    image = np.asarray(rgb_image).astype(np.uint8)
    cam = np.asarray(heatmap, dtype=np.float32)

    if image.shape != (IMG_SIZE, IMG_SIZE, 3):
        raise ValueError("Invalid original image dimensions.")

    if cam.shape != (IMG_SIZE, IMG_SIZE):
        raise ValueError("Invalid activation map dimensions.")

    cam = np.clip(cam, 0, 1)

    heatmap_bgr = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB
    )

    result = (
        0.55 * image.astype(np.float32)
        + 0.45 * heatmap_rgb.astype(np.float32)
    )

    return np.clip(result, 0, 255).astype(np.uint8)
