"""DP-CXR inference service — differentially private multimodal chest X-ray prediction."""

__version__ = "1.0.0"

from .predictor import DPCXRPredictor  # noqa: F401

__all__ = ["DPCXRPredictor"]
