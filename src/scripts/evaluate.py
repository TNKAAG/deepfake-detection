"""
06_evaluate.py
--------------
Evaluation script for all trained deepfake detection models.

Loads each model from src/models/, runs inference on the held-out
test set, and reports:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - AUC-ROC

Results are printed to console and saved to src/models/evaluation_results.csv.

Usage (from src/scripts/):
    python 06_evaluate.py

Requirements:
    pip install torch torchvision scikit-learn pandas tqdm timm
"""

import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from dataloaders import get_dataloaders, get_device
from baseline_cnn import get_baseline_model
from efficientnet import get_efficientnet
from xception import get_xception

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR       = WORKSPACE_ROOT / "src" / "data"
MODELS_DIR     = WORKSPACE_ROOT / "src" / "models"

BATCH_SIZE   = 32
NUM_WORKERS  = 4

# Registry of models to evaluate.
# Add Xception, FrequencyNet, ViT entries here as teammates finish training.
# Each entry: (display_name, checkpoint_filename, model_factory_function)
MODEL_REGISTRY = [
    (
        "Baseline CNN",
        "baseline_cnn_best.pth",
        lambda: get_baseline_model(),
    ),
    (
        "EfficientNet-B0",
        "efficientnet_phase2_best.pth",
        lambda: get_efficientnet(),
    ),
    (
        "Xception",
        "xception_best.pth",
        lambda: get_xception(),
    ),
    # ── Add remaining teammates' models below as they become available ──
    # (
    #     "Frequency CNN",
    #     "frequencycnn_best.pth",
    #     lambda: get_frequency_cnn(),  # import from frequency_cnn.py
    # ),
    # (
    #     "ViT-B/16",
    #     "vit_best.pth",
    #     lambda: get_vit(),            # import from vit_model.py
    # ),
]

# ─────────────────────────────────────────────


@torch.no_grad()
def run_inference(model, loader, device) -> tuple[list, list, list]:
    """
    Run inference on a dataloader.
    Returns (true_labels, predicted_labels, fake_class_probabilities).
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)

        # Class index for "fake" — ImageFolder assigns alphabetically: fake=0, real=1
        fake_idx = loader.dataset.class_to_idx["fake"]
        preds    = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs[:, fake_idx].cpu().tolist())

    return all_labels, all_preds, all_probs


def compute_metrics(
    true_labels: list,
    pred_labels: list,
    probs: list,
    class_to_idx: dict,
) -> dict:
    """
    Compute all evaluation metrics.
    Positive class = fake (what we care most about detecting).
    """
    fake_idx = class_to_idx["fake"]

    # Convert to binary: fake=1, real=0 for standard metric conventions
    binary_true = [1 if l == fake_idx else 0 for l in true_labels]
    binary_pred = [1 if p == fake_idx else 0 for p in pred_labels]

    acc       = accuracy_score(binary_true, binary_pred)
    precision = precision_score(binary_true, binary_pred, zero_division=0)
    recall    = recall_score(binary_true, binary_pred, zero_division=0)
    f1        = f1_score(binary_true, binary_pred, zero_division=0)

    try:
        auc = roc_auc_score(binary_true, probs)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(binary_true, binary_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1_score":  f1,
        "auc_roc":   auc,
        "TP": int(tp), "TN": int(tn),
        "FP": int(fp), "FN": int(fn),
    }


def evaluate_model(
    name:        str,
    ckpt_path:   Path,
    model_fn,
    test_loader,
    device:      torch.device,
) -> dict | None:
    """Load a checkpoint and evaluate on the test set."""

    if not ckpt_path.exists():
        print(f"\n  [SKIP] {name} — checkpoint not found: {ckpt_path.name}")
        return None

    print(f"\n{'─'*60}")
    print(f"Evaluating: {name}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"{'─'*60}")

    # Build model and load weights
    model = model_fn().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Run inference
    true_labels, pred_labels, probs = run_inference(model, test_loader, device)

    # Compute metrics
    metrics = compute_metrics(
        true_labels, pred_labels, probs,
        test_loader.dataset.class_to_idx
    )

    # Print results
    print(f"\n  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"\n  Confusion Matrix (positive = fake):")
    print(f"              Predicted")
    print(f"              Fake    Real")
    print(f"  Actual Fake  {metrics['TP']:>5}   {metrics['FN']:>5}")
    print(f"  Actual Real  {metrics['FP']:>5}   {metrics['TN']:>5}")

    metrics["model"] = name
    metrics["checkpoint"] = ckpt_path.name
    metrics["parameters"] = total_params
    return metrics


def print_comparison_table(results: list[dict]):
    """Print a clean side-by-side comparison of all evaluated models."""
    print(f"\n{'='*60}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print(f"{'─'*20} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for r in results:
        print(
            f"{r['model']:<20} "
            f"{r['accuracy']:>6.4f} "
            f"{r['precision']:>6.4f} "
            f"{r['recall']:>6.4f} "
            f"{r['f1_score']:>6.4f} "
            f"{r['auc_roc']:>6.4f}"
        )
    print(f"{'='*60}")


def main():
    device  = get_device()
    loaders = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Always evaluate on the test set only — never val
    test_loader = loaders["test"]
    print(f"\nTest set: {len(test_loader.dataset)} samples")

    all_results = []

    for name, ckpt_filename, model_fn in MODEL_REGISTRY:
        ckpt_path = MODELS_DIR / ckpt_filename
        metrics   = evaluate_model(name, ckpt_path, model_fn, test_loader, device)
        if metrics is not None:
            all_results.append(metrics)

    if not all_results:
        print("\nNo models were evaluated — check that checkpoints exist in src/models/")
        return

    # ── Comparison table ──────────────────────────────────────────
    print_comparison_table(all_results)

    # ── Save to CSV ───────────────────────────────────────────────
    out_path = MODELS_DIR / "evaluation_results.csv"
    cols     = ["model", "accuracy", "precision", "recall",
                "f1_score", "auc_roc", "TP", "TN", "FP", "FN",
                "parameters", "checkpoint"]
    df = pd.DataFrame(all_results)[cols]
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print("Next step: technical report.")


if __name__ == "__main__":
    main()
