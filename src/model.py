"""
Model definitions for Hand Gesture Recognition: Baseline (MobileNetV2) and SOTA (ConvNeXt-Tiny, Swin).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models


def create_mobilenetv2_baseline(num_classes: int = 3, pretrained: bool = False) -> nn.Module:
    """Create MobileNetV2 baseline matching Assignment 1 architecture."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def create_convnext_sota(
    num_classes: int = 3,
    pretrained: bool = True,
    variant: str = "tiny"
) -> nn.Module:
    """Create ConvNeXt SOTA classifier."""
    if variant == "tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif variant == "small":
        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = models.convnext_small(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported ConvNeXt variant: {variant}")
    return model


def get_target_layer_for_cam(model_name: str, model: nn.Module) -> nn.Module:
    """Return the designated feature layer for Grad-CAM heatmaps."""
    name = model_name.lower()
    if "mobilenet" in name:
        # Last conv block in MobileNetV2 features
        return model.features[-1]
    elif "convnext" in name:
        # Last stage in ConvNeXt features
        return model.features[-1]
    else:
        # Fallback to the last child in features if present
        if hasattr(model, "features"):
            return model.features[-1]
        raise ValueError(f"Cannot automatically determine target layer for Grad-CAM in model '{model_name}'.")


def load_model(
    model_name: str,
    weights_path: Optional[str] = None,
    num_classes: int = 3,
    device: str = "cpu"
) -> Tuple[nn.Module, nn.Module]:
    """
    Factory function to instantiate and load weights for a gesture model.
    Returns (model, target_cam_layer).
    """
    name = model_name.lower()
    if "mobilenet" in name:
        model = create_mobilenetv2_baseline(num_classes=num_classes, pretrained=(weights_path is None))
    elif "convnext" in name:
        model = create_convnext_sota(num_classes=num_classes, pretrained=(weights_path is None))
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=device)
        # If wrapped in a dictionary with 'state_dict' or 'model'
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    target_layer = get_target_layer_for_cam(model_name, model)
    return model, target_layer

