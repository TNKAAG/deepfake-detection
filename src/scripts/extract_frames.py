"""
02_extract_frames.py
--------------------
STEP 2 of 2 in the data pipeline.

Reads the manifest CSVs produced by 01_split_videos.py,
extracts frames from each video using decord,
and writes them to src/data/train/, src/data/val/, src/data/test/.

Frame filenames encode their source manipulation type so
per-manipulation evaluation is possible later:
    {source}_{video_id}_f{frame_number:06d}.jpg

Usage (from src/scripts/):
    python 02_extract_frames.py
    python 02_extract_frames.py --split train   # one split only
    python 02_extract_frames.py --dry-run       # no files written

Requirements:
    pip install decord pandas pillow tqdm
"""

import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    from decord import VideoReader, cpu
except ImportError:
    raise ImportError(
        "decord is not installed.\n"
        "Install it with:  pip install decord"
    )

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# workspace/src/scripts/ → up three levels to workspace/
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = WORKSPACE_ROOT / "src" / "data"

FRAMES_PER_VIDEO = 15
FRAME_SIZE       = (224, 224)
JPEG_QUALITY     = 95

# ─────────────────────────────────────────────


def extract_frames(
    video_path: Path,
    output_dir: Path,
    source: str,
    video_id: str,
    n_frames: int = FRAMES_PER_VIDEO,
    frame_size: tuple = FRAME_SIZE,
    dry_run: bool = False,
) -> int:
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
    except Exception as e:
        tqdm.write(f"  [WARN] Could not open {video_path.name}: {e}")
        return 0

    total_frames = len(vr)
    if total_frames == 0:
        tqdm.write(f"  [WARN] Zero frames in {video_path.name}")
        return 0

    n             = min(n_frames, total_frames)
    step          = max(1, total_frames // n)
    frame_indices = [i * step for i in range(n)]

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        frames = vr.get_batch(frame_indices).asnumpy()
    except Exception as e:
        tqdm.write(f"  [WARN] Failed to read frames from {video_path.name}: {e}")
        return 0

    safe_source = source.replace(" ", "_").replace("/", "_")
    written     = 0

    for frame_idx, frame_array in zip(frame_indices, frames):
        filename = f"{safe_source}_{video_id}_f{frame_idx:06d}.jpg"
        out_path = output_dir / filename

        if not dry_run:
            img = Image.fromarray(frame_array)
            img = img.resize(frame_size, Image.LANCZOS)
            img.save(str(out_path), "JPEG", quality=JPEG_QUALITY)

        written += 1

    return written


def process_manifest(manifest_path: Path, dry_run: bool = False) -> dict:
    split_name = manifest_path.stem.replace("_manifest", "")
    df         = pd.read_csv(manifest_path)
    stats      = {"total_videos": len(df), "total_frames": 0, "failed": 0}

    print(f"\n{'─'*60}")
    print(f"Processing split: {split_name.upper()}  ({len(df)} videos)")
    print(f"{'─'*60}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        video_path = Path(row["video_path"])
        label      = row["label"]
        source     = row["source"]
        video_id   = row["video_id"]
        output_dir = DATA_DIR / split_name / label

        if not video_path.exists():
            tqdm.write(f"  [WARN] Video not found: {video_path}")
            stats["failed"] += 1
            continue

        n_written = extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            source=source,
            video_id=video_id,
            dry_run=dry_run,
        )
        stats["total_frames"] += n_written
        if n_written == 0:
            stats["failed"] += 1

    return stats


def print_summary(all_stats: dict):
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    grand_frames, grand_failed = 0, 0
    for split, s in all_stats.items():
        print(f"  {split:<8}  videos={s['total_videos']:>5}  "
              f"frames={s['total_frames']:>7}  failed={s['failed']}")
        grand_frames += s["total_frames"]
        grand_failed += s["failed"]
    print(f"{'─'*60}")
    print(f"  {'TOTAL':<8}  frames={grand_frames:>7}  failed={grand_failed}")
    print(f"{'='*60}")


def count_existing_frames():
    print("\n── Current frame counts in src/data/ ───────────────────")
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            d     = DATA_DIR / split / label
            count = len(list(d.glob("*.jpg"))) if d.exists() else 0
            print(f"  {split}/{label}: {count:>6} frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    args = parser.parse_args()

    if args.dry_run:
        print("WARNING: DRY RUN — no files will be written.\n")

    manifests = sorted(DATA_DIR.glob("*_manifest.csv"))
    if not manifests:
        raise FileNotFoundError(
            f"No manifest CSVs found in {DATA_DIR}.\n"
            "Run 01_split_videos.py first."
        )

    if args.split:
        manifests = [m for m in manifests if args.split in m.name]

    all_stats = {}
    for manifest_path in manifests:
        split_name            = manifest_path.stem.replace("_manifest", "")
        all_stats[split_name] = process_manifest(manifest_path, dry_run=args.dry_run)

    print_summary(all_stats)

    if not args.dry_run:
        count_existing_frames()
        print(f"\nFrames written to: {DATA_DIR}")
        print("Next step: open notebooks/main.ipynb")


if __name__ == "__main__":
    main()
