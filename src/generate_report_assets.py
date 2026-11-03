"""
Generate report figures, confusion matrices, and Grad-CAM visual comparison.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.dataset import GestureDataset, get_default_transforms, CLASSES
from src.model import load_model
from src.evaluate import evaluate_model
from src.gradcam import GradCAM, apply_colormap_on_image

FIG_DIR = ROOT_DIR / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("[*] Generating report figures and comparison charts...")
    device = "cpu"

    # Paths
    mobilenet_weights = str(ROOT_DIR / "weights" / "baseline_mobilenet.pth")
    convnext_weights = str(ROOT_DIR / "weights" / "sota_convnext.pth")
    test_dir = str(ROOT_DIR / "data" / "split" / "test")

    # Load models
    m_base, cam_base = load_model("mobilenet", mobilenet_weights, len(CLASSES), device=device)
    m_sota, cam_sota = load_model("convnext", convnext_weights, len(CLASSES), device=device)

    # Test dataloader
    test_ds = GestureDataset(test_dir, transform=get_default_transforms(224, is_train=False))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # 1. Evaluate both models
    print("[*] Evaluating MobileNetV2...")
    res_base = evaluate_model(m_base, test_loader, device=device)

    print("[*] Evaluating ConvNeXt-Tiny...")
    res_sota = evaluate_model(m_sota, test_loader, device=device)

    # 2. Side-by-side Confusion Matrices
    print("[*] Plotting confusion matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    sns.heatmap(
        res_base["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0], cbar=False
    )
    axes[0].set_title(f"Baseline: MobileNetV2\nAccuracy: {res_base['accuracy']*100:.2f}% | F1: {res_base['macro_f1']:.4f}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Ground Truth")

    sns.heatmap(
        res_sota["confusion_matrix"], annot=True, fmt="d", cmap="Greens",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1], cbar=False
    )
    axes[1].set_title(f"SOTA: ConvNeXt-Tiny\nAccuracy: {res_sota['accuracy']*100:.2f}% | F1: {res_sota['macro_f1']:.4f}")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Ground Truth")

    plt.tight_layout()
    cm_path = FIG_DIR / "confusion_matrices_comparison.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] Saved {cm_path}")

    # 3. Grad-CAM Qualitative Samples
    print("[*] Generating Grad-CAM comparison samples...")
    sample_images = {}
    for cls in CLASSES:
        cls_folder = Path(test_dir) / cls
        imgs = list(cls_folder.glob("*.jpg")) + list(cls_folder.glob("*.png"))
        if imgs:
            sample_images[cls] = imgs[0]

    cam_ext_base = GradCAM(m_base, cam_base)
    cam_ext_sota = GradCAM(m_sota, cam_sota)

    transform = get_default_transforms(224, is_train=False)

    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(9, 3 * len(CLASSES)))

    for row_idx, cls in enumerate(CLASSES):
        img_path = sample_images[cls]
        raw_pil = Image.open(img_path).convert("RGB")
        input_t = transform(raw_pil).unsqueeze(0).to(device)

        # Baseline CAM
        hm_b, pred_b, conf_b, _ = cam_ext_base.generate_heatmap(input_t)
        over_b = apply_colormap_on_image(raw_pil, hm_b, alpha=0.55)

        # SOTA CAM
        hm_s, pred_s, conf_s, _ = cam_ext_sota.generate_heatmap(input_t)
        over_s = apply_colormap_on_image(raw_pil, hm_s, alpha=0.55)

        # Col 0: Original
        axes[row_idx, 0].imshow(raw_pil)
        axes[row_idx, 0].set_title(f"Target: {cls.upper()}")
        axes[row_idx, 0].axis("off")

        # Col 1: Baseline
        axes[row_idx, 1].imshow(over_b)
        axes[row_idx, 1].set_title(f"MobileNet: {CLASSES[pred_b]} ({conf_b*100:.1f}%)")
        axes[row_idx, 1].axis("off")

        # Col 2: SOTA
        axes[row_idx, 2].imshow(over_s)
        axes[row_idx, 2].set_title(f"ConvNeXt: {CLASSES[pred_s]} ({conf_s*100:.1f}%)")
        axes[row_idx, 2].axis("off")

    cam_ext_base.close()
    cam_ext_sota.close()

    plt.tight_layout()
    xai_path = FIG_DIR / "gradcam_comparison_samples.png"
    plt.savefig(xai_path, dpi=300)
    plt.close()
    print(f"[OK] Saved {xai_path}")

    # 4. Save JSON summary for the report
    summary = {
        "baseline_mobilenet": {
            "accuracy": res_base["accuracy"],
            "macro_f1": res_base["macro_f1"],
            "per_class": res_base["per_class"]
        },
        "sota_convnext": {
            "accuracy": res_sota["accuracy"],
            "macro_f1": res_sota["macro_f1"],
            "per_class": res_sota["per_class"]
        }
    }
    with open(ROOT_DIR / "report" / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] All report figures and metrics summary generated successfully!")


if __name__ == "__main__":
    main()

