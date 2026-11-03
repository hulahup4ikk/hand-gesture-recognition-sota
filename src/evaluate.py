"""
Evaluation and benchmark script: Accuracy, Macro-F1, Confusion Matrix, and Latency.
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import GestureDataset, get_default_transforms, CLASSES
from src.model import load_model


def benchmark_latency(
    model: torch.nn.Module,
    input_size: int = 224,
    device: str = "cpu",
    num_runs: int = 100,
    warmup: int = 15
) -> Dict[str, float]:
    """Measure inference latency in milliseconds and FPS."""
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    latencies_arr = np.array(latencies)
    mean_ms = float(np.mean(latencies_arr))
    p50_ms = float(np.percentile(latencies_arr, 50))
    p95_ms = float(np.percentile(latencies_arr, 95))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_ms, 2),
        "p50_latency_ms": round(p50_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "fps": round(fps, 1),
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Calculate classification metrics: accuracy, macro-F1, and confusion matrix."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.tolist())

    acc = float(accuracy_score(all_targets, all_preds))
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])

    per_prec, per_rec, per_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)

    per_class = {}
    for i, c in enumerate(CLASSES):
        per_class[c] = {
            "precision": round(float(per_prec[i]), 4),
            "recall": round(float(per_rec[i]), 4),
            "f1": round(float(per_f1[i]), 4),
        }

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "confusion_matrix": cm,
        "per_class": per_class,
        "all_preds": all_preds,
        "all_targets": all_targets,
    }


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str):
    """Plot and save confusion matrix figure."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_benchmark(
    model_name: str,
    weights_path: str,
    test_dir: str,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Run full evaluation and latency benchmark for a model."""
    model, _ = load_model(model_name, weights_path=weights_path, num_classes=len(CLASSES), device=device)
    test_ds = GestureDataset(test_dir, transform=get_default_transforms(224, is_train=False))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    metrics = evaluate_model(model, test_loader, device=device)
    latency = benchmark_latency(model, input_size=224, device=device)

    metrics.update(latency)
    metrics["device"] = device
    metrics["model_name"] = model_name
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate gesture model")
    parser.add_argument("--model", type=str, default="mobilenet", help="mobilenet or convnext")
    parser.add_argument("--weights", type=str, default="weights/baseline_mobilenet.pth")
    parser.add_argument("--test_dir", type=str, default="data/split/test")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    results = run_benchmark(args.model, args.weights, args.test_dir, args.device)
    print("=" * 50)
    print(f"Results for {args.model.upper()} on {args.test_dir}:")
    print(f"  Accuracy:  {results['accuracy'] * 100:.2f}%")
    print(f"  Macro-F1:  {results['macro_f1']:.4f}")
    print(f"  Latency:   {results['mean_latency_ms']:.2f} ms (FPS: {results['fps']})")
    print("  Per-class metrics:")
    for c, vals in results["per_class"].items():
        print(f"    {c:6s}: F1={vals['f1']:.4f}, Prec={vals['precision']:.4f}, Rec={vals['recall']:.4f}")
    print("=" * 50)

