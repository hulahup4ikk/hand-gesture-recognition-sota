"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for vision models.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, IDX_TO_CLASS


class GradCAM:
    """Native PyTorch Grad-CAM implementation."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_inp, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        """
        Generate Grad-CAM heatmap for the given input tensor (1, C, H, W).
        Returns:
            - heatmap: 2D numpy array [0..1]
            - predicted_class: int
            - confidence: float
            - all_probs: 1D numpy array of class probabilities
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]

        pred_idx = int(output.argmax(dim=1).item())
        if target_class_idx is None:
            target_class_idx = pred_idx

        # Backward pass for the target score
        target_score = output[0, target_class_idx]
        target_score.backward()

        # Global average pooling of gradients: weights alpha_k
        # Gradients shape: (1, Channels, H, W)
        grads = self.gradients
        activations = self.activations

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input resolution
        _, _, h, w = input_tensor.shape
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to 0..1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_idx, float(probs[pred_idx]), probs

    def close(self):
        """Remove hooks to prevent memory leaks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def apply_colormap_on_image(
    original_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay 2D heatmap [0..1] on a PIL Image using matplotlib colormap.
    Returns RGB uint8 image as numpy array (H, W, 3).
    """
    import matplotlib.pyplot as plt

    # Resize heatmap to original image dimensions
    w, h = original_img.size
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)
    norm_heatmap = np.array(heatmap_pil, dtype=np.float32) / 255.0

    # Colorize using 'jet' colormap
    colormap = plt.get_cmap("jet")
    colored_heatmap = colormap(norm_heatmap)[:, :, :3]  # drop alpha

    # Convert original to float
    orig_np = np.array(original_img).astype(np.float32) / 255.0

    # Blend
    blended = alpha * colored_heatmap + (1.0 - alpha) * orig_np
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return blended

