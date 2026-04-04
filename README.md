# CS464 — Deepfake Detection
**Group 14** | George · Nana · Shaun · Victor

Binary deepfake detection using the FaceForensics++ C23 dataset.
Compares a baseline CNN trained from scratch against a fine-tuned EfficientNet-B0.

---

## Repository Structure

```
workspace/
├── archive/                        ← NOT in git (videos too large)
│   └── FaceForensics++_C23/
│       ├── Deepfakes/
│       ├── DeepFakeDetection/
│       ├── Face2Face/
│       ├── FaceShifter/
│       ├── FaceSwap/
│       ├── NeuralTextures/
│       └── original/
└── src/
    ├── scripts/
    │   ├── 01_split_videos.py       split videos into train/val/test manifests
    │   ├── 02_extract_frames.py     extract frames from videos into src/data/
    │   ├── 03_dataloaders.py        PyTorch dataloaders + transforms
    │   ├── baseline_cnn.py          baseline CNN model definition
    │   ├── 04_train_baseline.py     baseline training loop
    │   ├── efficientnet.py          EfficientNet-B0 model  (coming soon)
    │   └── 05_train_efficientnet.py EfficientNet training loop (coming soon)
    ├── data/                        ← NOT in git (frames too large)
    │   ├── train_manifest.csv       video-level split manifests — ARE in git
    │   ├── val_manifest.csv
    │   ├── test_manifest.csv
    │   ├── train/ real/ fake/
    │   ├── val/   real/ fake/
    │   └── test/  real/ fake/
    ├── models/                      ← .pth files not in git
    │   └── baseline_training_log.csv
    └── notebooks/
        └── main.ipynb               summary notebook
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/deepfake-detection.git
cd deepfake-detection
```

### 2. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install decord pandas pillow scikit-learn tqdm
```

> **macOS (Apple Silicon):** Replace the torch install line with:
> ```bash
> pip install torch torchvision torchaudio
> ```

### 3. Get the dataset

Download the FaceForensics++ C23 dataset and place it at:
```
workspace/archive/FaceForensics++_C23/
```
The folder must contain: `Deepfakes/`, `DeepFakeDetection/`, `Face2Face/`,
`FaceShifter/`, `FaceSwap/`, `NeuralTextures/`, `original/`

> The dataset requires access approval. Request it at:
> https://github.com/ondyari/FaceForensics

### 4. Extract frames

The manifest CSVs (`train_manifest.csv` etc.) are already in `src/data/` from
the repo — they define the video-level train/val/test split.

Run the frame extractor to populate `src/data/train/`, `src/data/val/`, `src/data/test/`:

```bash
cd src/scripts
python 02_extract_frames.py
```

> This takes 30–60 minutes on the first run. Use `--split train` to do one split at a time.

### 5. Open the notebook

```bash
cd src/notebooks
jupyter notebook main.ipynb
```

Run **Section 0** first — it auto-detects the workspace root so paths work
on any machine regardless of where the repo is cloned.

---

## Hardware Notes

| Machine | Device | Notes |
|---------|--------|-------|
| Windows laptop (RTX 5050) | CUDA | Primary training machine |
| MacBook (Apple Silicon) | MPS | Data prep, analysis, visualisation |

The code auto-detects the available device — no changes needed when switching machines.

---

## Reproducing Results

```bash
# Step 1 — only needed if manifest CSVs are missing
python src/scripts/01_split_videos.py

# Step 2 — extract frames (skip if src/data/ already populated)
python src/scripts/02_extract_frames.py

# Step 3 — train baseline
python src/scripts/04_train_baseline.py

# Step 4 — train EfficientNet (coming soon)
# python src/scripts/05_train_efficientnet.py
```

All random seeds are fixed at `42` for reproducibility.

---

## Dependencies

```
torch >= 2.0
torchvision
decord
pandas
pillow
scikit-learn
tqdm
jupyter
```
