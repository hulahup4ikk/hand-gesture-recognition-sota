"""
Inference and Grad-CAM utility functions for the Streamlit demo.
"""

import time
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, CLASSES, IDX_TO_CLASS, get_default_transforms
from src.gradcam import GradCAM, apply_colormap_on_image


def preprocess_image(image: Image.Image, img_size: int = 224) -> Tuple[torch.Tensor, Image.Image]:
    """Preprocess PIL image for model input."""
    rgb_img = image.convert("RGB")
    transform = get_default_transforms(img_size=img_size, is_train=False)
    tensor = transform(rgb_img).unsqueeze(0)
    return tensor, rgb_img


def predict_and_explain(
    model: torch.nn.Module,
    target_cam_layer: torch.nn.Module,
    image: Image.Image,
    device: str = "cpu",
    cam_alpha: float = 0.55
) -> Dict[str, Any]:
    """
    Run model prediction and generate Grad-CAM explanation.
    Returns:
        - predicted_label: str
        - confidence: float
        - probabilities: Dict[str, float]
        - overlay_rgb: np.ndarray (H, W, 3)
        - latency_ms: float
    """
    tensor, rgb_img = preprocess_image(image)
    tensor = tensor.to(device)

    # Initialize GradCAM
    cam_extractor = GradCAM(model=model, target_layer=target_cam_layer)

    # Measure inference + XAI latency
    t0 = time.perf_counter()
    heatmap, pred_idx, confidence, probs = cam_extractor.generate_heatmap(tensor)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    # Clean up hooks
    cam_extractor.close()

    # Overlay heatmap on original image
    overlay_rgb = apply_colormap_on_image(rgb_img, heatmap, alpha=cam_alpha)

    # Format class probabilities
    probabilities = {
        CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))
    }

    return {
        "predicted_label": IDX_TO_CLASS[pred_idx],
        "confidence": confidence,
        "probabilities": probabilities,
        "overlay_rgb": overlay_rgb,
        "latency_ms": round(latency_ms, 2),
    }

