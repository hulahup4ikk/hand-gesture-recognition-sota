"""
Training pipeline for Hand Gesture Recognition (Baseline and SOTA).
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, Any, List
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.dataset import create_dataloaders, CLASSES
from src.model import load_model, create_convnext_sota, create_mobilenetv2_baseline
from src.evaluate import evaluate_model


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu"
) -> Dict[str, float]:
    """Train for a single epoch."""
    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return {"loss": avg_loss, "accuracy": acc}


def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str = "cpu"
) -> Dict[str, float]:
    """Validate model performance."""
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return {"loss": avg_loss, "accuracy": acc}


def train_model(
    model_name: str = "convnext",
    data_dir: str = "data/split",
    num_epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cpu",
    save_path: str = "weights/sota_convnext.pth"
) -> Dict[str, Any]:
    """Full training pipeline."""
    print(f"[*] Starting training: Model={model_name.upper()}, Device={device}, Epochs={num_epochs}")
    train_loader, val_loader, test_loader = create_dataloaders(data_dir, batch_size=batch_size)

    if "mobilenet" in model_name.lower():
        model = create_mobilenetv2_baseline(num_classes=len(CLASSES), pretrained=True)
    elif "convnext" in model_name.lower():
        model = create_convnext_sota(num_classes=len(CLASSES), pretrained=True, variant="tiny")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    best_val_acc = 0.0
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for epoch in range(1, num_epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device=device)
        val = validate(model, val_loader, criterion, device=device)
        scheduler.step()

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["accuracy"])
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {tr['loss']:.4f} | Train Acc: {tr['accuracy']*100:.1f}% | "
            f"Val Loss: {val['loss']:.4f} | Val Acc: {val['accuracy']*100:.1f}%"
        )

        if val["accuracy"] >= best_val_acc:
            best_val_acc = val["accuracy"]
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val Acc: {best_val_acc*100:.2f}%)")

    total_time = time.time() - t_start
    print(f"\n[OK] Training completed in {total_time/60:.2f} min! Best Val Acc: {best_val_acc*100:.2f}%")

    # Evaluate best model on test set
    print("[*] Evaluating best model on test set...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_metrics = evaluate_model(model, test_loader, device=device)

    print("=" * 50)
    print(f"TEST BENCHMARK RESULTS ({model_name.upper()}):")
    print(f"  Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"  Macro-F1:      {test_metrics['macro_f1']:.4f}")
    print("=" * 50)

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"{model_name.upper()} Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title(f"{model_name.upper()} Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    fig_dir = Path("report/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"training_curves_{model_name}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[OK] Training curves saved to {fig_path}")

    return {
        "best_val_acc": best_val_acc,
        "test_metrics": test_metrics,
        "history": history
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gesture model")
    parser.add_argument("--model", type=str, default="convnext", help="convnext or mobilenet")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default="weights/sota_convnext.pth")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=args.save_path
    )

