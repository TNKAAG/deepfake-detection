"""
train_vit.py
------------
Two-phase training loop for ViT-B/16 on FaceForensics++ frames.

Phase 1 — backbone frozen, head only (faster, avoids catastrophic forgetting):
  Epochs : PHASE1_EPOCHS  (default 10)
  LR     : PHASE1_LR      (default 1e-3)

Phase 2 — full fine-tuning with a very small LR:
  Epochs : PHASE2_EPOCHS  (default 15)
  LR     : PHASE2_LR      (default 1e-5)

Outputs to src/models/:
  vit_phase1_best.pth       best checkpoint from Phase 1
  vit_phase2_best.pth       best checkpoint from Phase 2  ← use for evaluation
  vit_final.pth             weights at end of training
  vit_training_log.csv      per-epoch metrics (includes 'phase' column)

Usage (from src/scripts/):
    python train_vit.py
    python train_vit.py --start-phase 2   # skip Phase 1, load phase1 best

Requirements:
    pip install torch torchvision scikit-learn pandas tqdm
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from vit_model import get_vit_model, freeze_backbone, unfreeze_backbone
from dataloaders import get_dataloaders, get_device, sanity_check

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR   = WORKSPACE_ROOT / "data"
MODELS_DIR = WORKSPACE_ROOT / "src" / "models"

BATCH_SIZE   = 16    # ViT-B/16 is 86 M params — 16 is safe on 8.5 GB VRAM
NUM_WORKERS  = 4

PHASE1_EPOCHS = 10
PHASE1_LR     = 1e-1

PHASE2_EPOCHS = 15
PHASE2_LR     = 1e-5

LR_PATIENCE        = 3
LR_FACTOR          = 0.5
MIN_LR             = 1e-7
EARLY_STOP_PATIENCE = 5

WEIGHT_DECAY = 1e-4

# ─────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
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


def log_epoch(log_path, row, write_header=False):
    mode = "w" if write_header else "a"
    with open(log_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_phase(
    phase_num,
    model,
    loaders,
    criterion,
    device,
    scaler,
    num_epochs,
    lr,
    log_path,
    first_log_row,
    ckpt_name,
):
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=MIN_LR,
    )

    best_val_loss = float("inf")
    no_improve    = 0

    print("=" * 60)
    print(f"Phase {phase_num}  |  epochs={num_epochs}  lr={lr:.0e}  batch={BATCH_SIZE}")
    print("=" * 60)

    for epoch in range(num_epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        elapsed     = time.time() - t0
        current_lr  = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch+1:>2}/{num_epochs} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"lr={current_lr:.2e}  ({elapsed:.0f}s)"
        )

        log_epoch(log_path, {
            "phase":      phase_num,
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc, 6),
            "val_loss":   round(val_metrics["loss"], 6),
            "val_acc":    round(val_metrics["accuracy"], 6),
            "val_auc":    round(val_metrics["auc"], 6),
            "lr":         current_lr,
            "elapsed_s":  round(elapsed, 1),
        }, write_header=first_log_row[0])
        first_log_row[0] = False

        scheduler.step(val_metrics["loss"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            no_improve    = 0
            save_checkpoint(
                model,
                MODELS_DIR / ckpt_name,
                {"phase": phase_num, "epoch": epoch, "val_loss": best_val_loss,
                 "val_auc": val_metrics["auc"]},
            )
            print(
                f"  Checkpoint saved → {ckpt_name}\n"
                f"  ✓ New best  (val_loss={best_val_loss:.4f}  val_auc={val_metrics['auc']:.4f})"
            )
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping — no improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    print(f"\nPhase {phase_num} complete.")
    print(f"  Best val_loss={best_val_loss:.4f}")
    return best_val_loss


def train(start_phase: int = 1):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device  = get_device()
    loaders = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    sanity_check(loaders)

    model = get_vit_model().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: ViT-B/16  ({total_params:,} total parameters)\n")

    class_weights = loaders["train"].class_weights.to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    log_path      = MODELS_DIR / "vit_training_log.csv"
    first_log_row = [True]   # mutable flag so run_phase can flip it

    # ── Phase 1: frozen backbone ──────────────────────────────────
    if start_phase == 1:
        freeze_backbone(model)
        run_phase(
            phase_num=1, model=model, loaders=loaders, criterion=criterion,
            device=device, scaler=scaler, num_epochs=PHASE1_EPOCHS, lr=PHASE1_LR,
            log_path=log_path, first_log_row=first_log_row,
            ckpt_name="vit_phase1_best.pth",
        )

    # ── Load best Phase-1 weights before Phase 2 ─────────────────
    phase1_ckpt = MODELS_DIR / "vit_phase1_best.pth"
    if phase1_ckpt.exists():
        ckpt = torch.load(phase1_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("\nLoaded Phase 1 best checkpoint for Phase 2.")
    else:
        print("\nNo Phase 1 checkpoint found — starting Phase 2 from current weights.")

    # ── Phase 2: full fine-tuning ─────────────────────────────────
    unfreeze_backbone(model)
    run_phase(
        phase_num=2, model=model, loaders=loaders, criterion=criterion,
        device=device, scaler=scaler, num_epochs=PHASE2_EPOCHS, lr=PHASE2_LR,
        log_path=log_path, first_log_row=first_log_row,
        ckpt_name="vit_phase2_best.pth",
    )

    save_checkpoint(
        model,
        MODELS_DIR / "vit_final.pth",
        {"phase": 2, "epoch": PHASE1_EPOCHS + PHASE2_EPOCHS},
    )

    print("\n" + "=" * 60)
    print("ViT-B/16 training complete.")
    print(f"  Best checkpoint : {MODELS_DIR / 'vit_phase2_best.pth'}")
    print(f"  Training log    : {log_path}")
    print("=" * 60)
    print("Next step: run evaluation to compare all three models on the test set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-phase", type=int, choices=[1, 2], default=1,
        help="Start from phase 1 (default) or skip to phase 2.",
    )
    args = parser.parse_args()
    train(start_phase=args.start_phase)
