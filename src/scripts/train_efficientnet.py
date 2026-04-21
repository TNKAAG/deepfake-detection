"""
05_train_efficientnet.py
------------------------
Two-phase training loop for EfficientNet-B0 deepfake detection.

Phase 1 — Frozen backbone (head only):
  Higher LR (1e-3), more aggressive training.
  Runs for PHASE1_EPOCHS epochs or until early stopping.

Phase 2 — Full fine-tuning:
  Very low LR (1e-5) to preserve pretrained features.
  Loads the best Phase 1 checkpoint before starting.
  Runs for PHASE2_EPOCHS epochs or until early stopping.

Outputs to src/models/:
  efficientnet_phase1_best.pth      best Phase 1 checkpoint
  efficientnet_phase2_best.pth      best Phase 2 checkpoint  ← use this for evaluation
  efficientnet_training_log.csv     combined per-epoch metrics

Usage (from src/scripts/):
    python 05_train_efficientnet.py
    python 05_train_efficientnet.py --phase 2   # skip Phase 1, jump straight to Phase 2

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

from efficientnet import get_efficientnet, freeze_backbone, unfreeze_backbone
from dataloaders import get_dataloaders, get_device, sanity_check

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR   = WORKSPACE_ROOT / "src" / "data"
MODELS_DIR = WORKSPACE_ROOT / "src" / "models"

# ── Phase 1 — head only ───────────────────────────────────────────
PHASE1_EPOCHS    = 10      # head converges quickly — 10 is usually enough
PHASE1_LR        = 1e-3    # standard Adam LR for training from scratch
PHASE1_PATIENCE  = 4       # early stop if no improvement for 4 epochs

# ── Phase 2 — full fine-tune ──────────────────────────────────────
PHASE2_EPOCHS    = 15      # more epochs — backbone needs gentle nudging
PHASE2_LR        = 1e-5    # CRITICAL: must be much lower than Phase 1
                            # too high → corrupts pretrained features
PHASE2_PATIENCE  = 5

# ── Shared ────────────────────────────────────────────────────────
BATCH_SIZE   = 32
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4       # set to 0 if Windows multiprocessing errors occur
LR_FACTOR    = 0.5
MIN_LR       = 1e-7

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

    return {"loss": running_loss / total, "accuracy": correct / total, "auc": auc}


def save_checkpoint(model, path, metadata):
    torch.save({"model_state_dict": model.state_dict(), **metadata}, path)
    print(f"  Checkpoint saved → {path.name}")


def log_epoch(log_path, row, write_header=False):
    mode = "w" if write_header else "a"
    with open(log_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────
# PHASE RUNNERS
# ─────────────────────────────────────────────

def run_phase(
    phase:        int,
    model:        nn.Module,
    loaders:      dict,
    criterion:    nn.Module,
    device:       torch.device,
    scaler,
    num_epochs:   int,
    lr:           float,
    patience:     int,
    log_path:     Path,
    first_phase:  bool,
) -> dict:
    """
    Generic training phase — runs for num_epochs with early stopping.
    Returns the best val metrics achieved during this phase.
    """
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    # filter() ensures Phase 1 optimiser only touches head parameters —
    # frozen backbone params are excluded automatically.

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR,
        patience=max(1, patience // 2), min_lr=MIN_LR,
    )

    ckpt_path     = MODELS_DIR / f"efficientnet_phase{phase}_best.pth"
    best_val_loss = float("inf")
    best_metrics  = {}
    no_improve    = 0
    val_metrics   = {}

    print(f"\n{'='*60}")
    print(f"Phase {phase}  |  epochs={num_epochs}  lr={lr:.0e}  "
          f"batch={BATCH_SIZE}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        elapsed     = time.time() - t0
        lr_curr     = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch+1:>2}/{num_epochs} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"lr={lr_curr:.2e}  ({elapsed:.0f}s)"
        )

        # Write header only for very first epoch of the whole run
        write_header = first_phase and (epoch == 0)
        log_epoch(log_path, {
            "phase":      phase,
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc, 6),
            "val_loss":   round(val_metrics["loss"], 6),
            "val_acc":    round(val_metrics["accuracy"], 6),
            "val_auc":    round(val_metrics["auc"], 6),
            "lr":         lr_curr,
            "elapsed_s":  round(elapsed, 1),
        }, write_header=write_header)

        scheduler.step(val_metrics["loss"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics  = val_metrics.copy()
            no_improve    = 0
            save_checkpoint(model, ckpt_path, {
                "phase":    phase,
                "epoch":    epoch + 1,
                "val_loss": best_val_loss,
                "val_auc":  val_metrics["auc"],
            })
            print(f"  ✓ New best  (val_loss={best_val_loss:.4f}  "
                  f"val_auc={val_metrics['auc']:.4f})")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\n  Early stopping — no improvement for {patience} epochs.")
            break

    return best_metrics


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def train(start_phase: int = 1):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device  = get_device()
    loaders = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    sanity_check(loaders)

    model = get_efficientnet().to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"\nModel: EfficientNet-B0  ({total:,} total parameters)")

    class_weights = loaders["train"].class_weights.to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    scaler   = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
    log_path = MODELS_DIR / "efficientnet_training_log.csv"

    # ── Phase 1 — frozen backbone ─────────────────────────────────
    if start_phase <= 1:
        freeze_backbone(model)

        phase1_best = run_phase(
            phase       = 1,
            model       = model,
            loaders     = loaders,
            criterion   = criterion,
            device      = device,
            scaler      = scaler,
            num_epochs  = PHASE1_EPOCHS,
            lr          = PHASE1_LR,
            patience    = PHASE1_PATIENCE,
            log_path    = log_path,
            first_phase = True,
        )

        print(f"\nPhase 1 complete.")
        print(f"  Best val_loss={phase1_best['loss']:.4f}  "
              f"val_auc={phase1_best['auc']:.4f}")

    # ── Phase 2 — full fine-tune ──────────────────────────────────
    if start_phase <= 2:
        # Always reload the best Phase 1 checkpoint before Phase 2
        # — ensures Phase 2 starts from the best head, not an overfit one
        p1_ckpt = MODELS_DIR / "efficientnet_phase1_best.pth"
        if p1_ckpt.exists():
            ckpt = torch.load(p1_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"\nLoaded Phase 1 best checkpoint for Phase 2.")
        else:
            print(f"\n[WARN] No Phase 1 checkpoint found — starting Phase 2 from current weights.")

        unfreeze_backbone(model)

        phase2_best = run_phase(
            phase       = 2,
            model       = model,
            loaders     = loaders,
            criterion   = criterion,
            device      = device,
            scaler      = scaler,
            num_epochs  = PHASE2_EPOCHS,
            lr          = PHASE2_LR,
            patience    = PHASE2_PATIENCE,
            log_path    = log_path,
            first_phase = (start_phase == 2),  # write header only if skipped Phase 1
        )

        print(f"\nPhase 2 complete.")
        print(f"  Best val_loss={phase2_best['loss']:.4f}  "
              f"val_auc={phase2_best['auc']:.4f}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EfficientNet training complete.")
    print(f"  Use for evaluation: efficientnet_phase2_best.pth")
    print(f"  Training log:       {log_path}")
    print(f"{'='*60}")
    print("\nNext step: run 06_evaluate.py to compare both models on the test set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=1,
        help="Start from phase 1 (default) or skip to phase 2."
    )
    args = parser.parse_args()
    train(start_phase=args.phase)
