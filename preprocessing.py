
"""Image decoding and preprocessing matching notebook Section 4."""

import cv2
import numpy as np

from src.config import IMG_SIZE


class InvalidImageError(ValueError):
    """Raised when an uploaded image cannot be decoded or processed."""


def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode uploaded image bytes into an OpenCV BGR image.
    """
    if not image_bytes:
        raise InvalidImageError("The uploaded image is empty.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if image is None:
        raise InvalidImageError(
            "Could not read this image. Please upload a valid image file."
        )

    return image


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """
    Apply the exact preprocessing pipeline used in notebook Section 4.

    Steps:
      1. BGR to RGB
      2. Resize to 224x224 using INTER_AREA
      3. Median blur with kernel size 3
      4. RGB to YCrCb
      5. Histogram equalization on Y channel only
      6. Convert YCrCb back to RGB

    Returns:
        uint8 RGB image of shape (224, 224, 3).
        Pixel values remain in the 0-255 range.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise InvalidImageError("Invalid image data.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise InvalidImageError("Expected a three-channel color image.")

    if image_bgr.dtype != np.uint8:
        raise InvalidImageError("Expected an 8-bit color image.")

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    image = cv2.medianBlur(image, 3)

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])

    processed = cv2.cvtColor(
        ycrcb,
        cv2.COLOR_YCrCb2RGB
    )

    return processed
