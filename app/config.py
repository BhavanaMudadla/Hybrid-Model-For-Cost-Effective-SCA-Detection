
"""Shared configuration for the Sickle Cell research prototype."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "final_hybrid_model.keras"

# Model input configuration
IMG_SIZE = 224
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
THRESHOLD = 0.5

# Notebook Grad-CAM configuration
EFFNET_LAYER = "efficientnetb0"
GRADCAM_LAYER = "top_activation"

# Supported image formats
ALLOWED_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
