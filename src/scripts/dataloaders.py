"""
03_dataloaders.py
-----------------
Dataloader setup for the FaceForensics++ deepfake detection project.

Provides:
  - EfficientNet-B0-compatible transforms (train + val/test)
  - WeightedRandomSampler to handle real/fake class imbalance
  - Class weights tensor for use with CrossEntropyLoss
  - Auto device detection: CUDA → MPS → CPU
  - Sanity check utility

Usage (import into notebook or other scripts):
    import sys
    sys.path.insert(0, '../scripts')   # from notebooks/
    from dataloaders import get_dataloaders, get_device, sanity_check

Requirements:
    pip install torch torchvision pillow
"""

import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# workspace/src/scripts/ → up three levels to workspace/
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = WORKSPACE_ROOT / "data"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Auto-detects the best available device.
      CUDA  → Windows laptop with RTX 5050
      MPS   → Apple Silicon MacBooks
      CPU   → fallback
    The same code runs unchanged on all four team machines.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[device] CUDA — {name}  ({mem_gb:.1f} GB VRAM)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[device] MPS — Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("[device] CPU — no GPU found")
    return device


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────

def get_transforms() -> dict:
    """
    Returns train and eval transform pipelines.

    Augmentation rationale for deepfake detection:
      - HorizontalFlip: faces are symmetric; flipping doesn't change authenticity
      - ColorJitter:    simulates codec/compression colour shifts common in deepfakes
      - RandomRotation: small rotations add robustness without distorting facial geometry

    Val/test: resize + normalize only. Never augment evaluation data.
    """
    train_transforms = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transforms = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {"train": train_transforms, "eval": eval_transforms}


# ─────────────────────────────────────────────
# CLASS WEIGHTS + SAMPLER
# ─────────────────────────────────────────────

def compute_class_weights(dataset: ImageFolder) -> torch.Tensor:
    """
    Inverse-frequency class weights for CrossEntropyLoss.
    Corrects for the ~1:5 real/fake imbalance in FaceForensics++.
    """
    counts   = Counter(dataset.targets)
    n_total  = len(dataset)
    n_classes = len(dataset.classes)

    weights = torch.zeros(n_classes)
    for class_idx, count in counts.items():
        weights[class_idx] = n_total / (n_classes * count)
    return weights


def make_weighted_sampler(dataset: ImageFolder) -> WeightedRandomSampler:
    """
    WeightedRandomSampler for balanced training batches.
    Used together with class-weighted loss for best imbalance handling.
    """
    class_weights  = compute_class_weights(dataset)
    sample_weights = torch.tensor(
        [class_weights[label] for label in dataset.targets], dtype=torch.float
    )
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


# ─────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────

def get_dataloaders(
    data_dir: str | Path = DATA_DIR,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """
    Build train / val / test DataLoaders.

    Args:
        data_dir:    Path to src/data/ — must contain train/, val/, test/.
        batch_size:  32 is safe for RTX 5050 at 224x224.
                     Increase to 64 if VRAM allows.
        num_workers: Set to 0 if multiprocessing errors occur on Windows.

    Returns:
        dict with keys "train", "val", "test".
        train loader also carries .class_weights and .class_to_idx attributes.
    """
    data_dir   = Path(data_dir)
    transforms = get_transforms()

    train_dataset = ImageFolder(data_dir / "train", transform=transforms["train"])
    val_dataset   = ImageFolder(data_dir / "val",   transform=transforms["eval"])
    test_dataset  = ImageFolder(data_dir / "test",  transform=transforms["eval"])

    assert train_dataset.class_to_idx == val_dataset.class_to_idx == test_dataset.class_to_idx, \
        "Class indices differ between splits — check folder structure."

    class_weights = compute_class_weights(train_dataset)
    sampler       = make_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,          # handles ordering — do not also set shuffle=True
        num_workers=num_workers,
        pin_memory=True,          # speeds up CPU→GPU transfers on the RTX 5050
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Attach extras for easy notebook access
    train_loader.class_weights = class_weights
    train_loader.class_to_idx  = train_dataset.class_to_idx

    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────

def sanity_check(loaders: dict):
    """
    Print class distributions, weights, and a sample batch shape.
    Always run this before starting a training run.
    """
    print("=" * 60)
    print("DATALOADER SANITY CHECK")
    print("=" * 60)

    for split, loader in loaders.items():
        dataset = loader.dataset
        counts  = Counter(dataset.targets)
        total   = len(dataset)

        print(f"\n  [{split.upper()}]  {total} samples")
        for class_name, idx in sorted(dataset.class_to_idx.items()):
            n   = counts[idx]
            pct = 100 * n / total
            bar = "█" * int(pct / 2)
            print(f"    {class_name:<6} (idx={idx}):  {n:>7} samples  {pct:>5.1f}%  {bar}")

    cw = loaders["train"].class_weights
    print(f"\n  Class weights (for CrossEntropyLoss):")
    for class_name, idx in sorted(loaders["train"].class_to_idx.items()):
        print(f"    {class_name:<6} (idx={idx}):  weight = {cw[idx]:.4f}")

    print(f"\n  Fetching one sample batch from train loader...")
    images, labels = next(iter(loaders["train"]))
    ci = loaders["train"].class_to_idx
    print(f"    images shape  : {tuple(images.shape)}")
    print(f"    labels shape  : {tuple(labels.shape)}")
    print(f"    pixel range   : [{images.min():.3f}, {images.max():.3f}]")
    print(f"    batch fake    : {(labels == ci['fake']).sum().item()}")
    print(f"    batch real    : {(labels == ci['real']).sum().item()}")
    print("\n" + "=" * 60)
    print("Sanity check passed. Ready to train.")
    print("=" * 60)


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device  = get_device()
    loaders = get_dataloaders()
    sanity_check(loaders)
