"""
01_split_videos.py
------------------
STEP 1 of 2 in the data pipeline.

Reads all video files from the FaceForensics++_C23 archive folder,
performs a VIDEO-LEVEL train/val/test split (70/15/15),
and writes three manifest CSV files to src/data/.

Run this ONCE. The manifests become your permanent source of truth.
Do not re-run unless you want to redo the entire split.

Usage (from src/scripts/):
    python 01_split_videos.py

Requirements:
    pip install pandas scikit-learn
"""

import os
import random
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# This script lives at: workspace/src/scripts/01_split_videos.py
# So:  parent       = src/scripts/
#      parent.parent = src/
#      parent.parent.parent = workspace/
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

ARCHIVE_DIR = WORKSPACE_ROOT / "archive" / "FaceForensics++_C23"
OUTPUT_DIR  = WORKSPACE_ROOT / "src" / "data"

FAKE_FOLDERS = [
    "Deepfakes",
    "DeepFakeDetection",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]

REAL_FOLDER = "original"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
RANDOM_SEED = 42

# ─────────────────────────────────────────────


def find_videos(folder: Path) -> list[Path]:
    return [
        p for p in folder.rglob("*")
        if p.suffix.lower() in VIDEO_EXTENSIONS
    ]


def split_video_list(videos: list[Path], seed: int = RANDOM_SEED):
    train, temp = train_test_split(videos, test_size=(1 - TRAIN_RATIO), random_state=seed)
    val, test   = train_test_split(temp,   test_size=0.5,               random_state=seed)
    return train, val, test


def build_records(videos: list[Path], label: str, split: str, source: str) -> list[dict]:
    return [
        {
            "video_path": str(v),
            "label":      label,
            "source":     source,
            "split":      split,
            "video_id":   v.stem,
        }
        for v in videos
    ]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)

    all_records = []

    # ── Real videos ──────────────────────────────────────────────
    real_dir = ARCHIVE_DIR / REAL_FOLDER
    assert real_dir.exists(), f"Cannot find real folder: {real_dir}"

    real_videos = find_videos(real_dir)
    assert len(real_videos) > 0, f"No videos found in {real_dir}"

    train_r, val_r, test_r = split_video_list(real_videos)
    all_records += build_records(train_r, "real", "train", "original")
    all_records += build_records(val_r,   "real", "val",   "original")
    all_records += build_records(test_r,  "real", "test",  "original")

    print(f"[real]  total={len(real_videos):>5}  "
          f"train={len(train_r)}  val={len(val_r)}  test={len(test_r)}")

    # ── Fake videos ───────────────────────────────────────────────
    for folder_name in FAKE_FOLDERS:
        fake_dir = ARCHIVE_DIR / folder_name
        if not fake_dir.exists():
            print(f"[WARN]  Folder not found, skipping: {fake_dir}")
            continue

        fake_videos = find_videos(fake_dir)
        if len(fake_videos) == 0:
            print(f"[WARN]  No videos found in {fake_dir}, skipping.")
            continue

        train_f, val_f, test_f = split_video_list(fake_videos)
        all_records += build_records(train_f, "fake", "train", folder_name)
        all_records += build_records(val_f,   "fake", "val",   folder_name)
        all_records += build_records(test_f,  "fake", "test",  folder_name)

        print(f"[{folder_name:<20}]  total={len(fake_videos):>5}  "
              f"train={len(train_f)}  val={len(val_f)}  test={len(test_f)}")

    # ── Write manifests ───────────────────────────────────────────
    df = pd.DataFrame(all_records)

    for split_name in ["train", "val", "test"]:
        subset   = df[df["split"] == split_name]
        out_path = OUTPUT_DIR / f"{split_name}_manifest.csv"
        subset.to_csv(out_path, index=False)
        print(f"\nSaved {split_name}_manifest.csv  ({len(subset)} videos)")

    print("\n── Class balance per split ──────────────────────────────")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())

    print("\n── Per manipulation type ────────────────────────────────")
    print(df.groupby(["source", "split"]).size().unstack(fill_value=0).to_string())

    print(f"\nManifests written to: {OUTPUT_DIR}")
    print("Next step: run  02_extract_frames.py")


if __name__ == "__main__":
    main()
