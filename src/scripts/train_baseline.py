"""
04_train_baseline.py
--------------------
Training loop for the baseline CNN.

Trains BaselineCNN from scratch on FaceForensics++ frames,
evaluating on the validation set after every epoch.

Outputs to src/models/:
  baseline_cnn_best.pth       best checkpoint (lowest val loss)
  baseline_cnn_final.pth      final epoch weights
  baseline_training_log.csv   per-epoch metrics

Usage (from src/scripts/):
    python 04_train_baseline.py
    python 04_train_baseline.py --resume ../models/baseline_cnn_best.pth

Requirements:
    pip install torch torchvision scikit-learn pandas tqdm
"""

import argparse
import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from baseline_cnn import get_baseline_model
from dataloaders import get_dataloaders, get_device, sanity_check

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# workspace/src/scripts/ → up three levels to workspace/
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR   = WORKSPACE_ROOT / "src" / "data"
MODELS_DIR = WORKSPACE_ROOT / "src" / "models"

NUM_EPOCHS    = 20
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 4   # set to 0 if Windows multiprocessing errors occur

LR_PATIENCE        = 3
LR_FACTOR          = 0.5
MIN_LR             = 1e-6
EARLY_STOP_PATIENCE = 7

# ─────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct      += (outputs.argmax(dim=1) == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, split="val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels        = [], []

    for images, labels in tqdm(loader, desc=f"  {split}", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs      = model(images)
        loss         = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct      += (outputs.argmax(dim=1) == labels).sum().item()
        total        += labels.size(0)

        probs = torch.softmax(outputs, dim=1)[:, 0].cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().tolist())

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return {
        "loss":     running_loss / total,
        "accuracy": correct / total,
        "auc":      auc,
    }


def save_checkpoint(model, path, metadata):
    torch.save({"model_state_dict": model.state_dict(), **metadata}, path)


def log_epoch(log_path, row, write_header=False):
    mode = "w" if write_header else "a"
    with open(log_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(resume_path=None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device  = get_device()
    loaders = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    sanity_check(loaders)

    model        = get_baseline_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: BaselineCNN  ({total_params:,} parameters)\n")

    class_weights = loaders["train"].class_weights.to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=MIN_LR,
    )

    # Mixed precision — ~2x faster on CUDA, ignored on MPS/CPU
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    start_epoch   = 0
    best_val_loss = float("inf")
    no_improve    = 0
    val_metrics   = {}

    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from {resume_path}  (epoch {start_epoch})\n")

    log_path = MODELS_DIR / "baseline_training_log.csv"

    print("=" * 60)
    print(f"Training BaselineCNN  |  {NUM_EPOCHS} epochs  |  batch={BATCH_SIZE}")
    print("=" * 60)

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        elapsed     = time.time() - t0
        lr          = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:>3}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"lr={lr:.2e}  ({elapsed:.0f}s)"
        )

        log_epoch(log_path, {
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc, 6),
            "val_loss":   round(val_metrics["loss"], 6),
            "val_acc":    round(val_metrics["accuracy"], 6),
            "val_auc":    round(val_metrics["auc"], 6),
            "lr":         lr,
            "elapsed_s":  round(elapsed, 1),
        }, write_header=(epoch == start_epoch))

        scheduler.step(val_metrics["loss"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            no_improve    = 0
            save_checkpoint(
                model,
                MODELS_DIR / "baseline_cnn_best.pth",
                {"epoch": epoch, "val_loss": best_val_loss, "val_auc": val_metrics["auc"]},
            )
            print(f"  New best model saved  (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping — no improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    save_checkpoint(
        model,
        MODELS_DIR / "baseline_cnn_final.pth",
        {"epoch": NUM_EPOCHS, "val_loss": val_metrics.get("loss", 0)},
    )

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Best checkpoint : {MODELS_DIR / 'baseline_cnn_best.pth'}")
    print(f"  Training log    : {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    train(resume_path=args.resume)