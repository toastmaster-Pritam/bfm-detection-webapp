#!/usr/bin/env python3
"""
Download pretrained YOLOv8 weights for the flagellar motor detection webapp.

Three methods are tried in order:
  1. Kaggle API  – downloads the output of a public YOLO training notebook
  2. gdown       – downloads the MIC-DKFZ nnU-Net checkpoint from Google Drive
  3. Manual      – prints step-by-step instructions

Usage:
    python download_weights.py              # tries all methods
    python download_weights.py --kaggle     # force Kaggle-only
    python download_weights.py --manual     # just print instructions
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# ───────────────────────────────────────────────────────────────────
# Method 1: Kaggle API (best for YOLOv8 – matches thesis pipeline)
# ───────────────────────────────────────────────────────────────────

KAGGLE_NOTEBOOK = "ravaghi/flagellar-motor-detection-2-3-yolo-training"


def try_kaggle():
    """Download YOLO weights from a public Kaggle notebook output."""
    print("\n[Method 1] Trying Kaggle API...")

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(f"  ✗ No Kaggle API key found at {kaggle_json}")
        print("    To set up:")
        print("    1. Go to https://www.kaggle.com/settings → API → Create New Token")
        print("    2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("    3. chmod 600 ~/.kaggle/kaggle.json")
        print("    4. Re-run this script")
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("  ✗ kaggle package not installed (pip install kaggle)")
        return False

    try:
        api = KaggleApi()
        api.authenticate()

        out_dir = WEIGHTS_DIR / "_kaggle_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading output of {KAGGLE_NOTEBOOK}...")
        api.kernels_output(KAGGLE_NOTEBOOK, path=str(out_dir))

        # Look for best.pt in the downloaded files
        for pt_file in out_dir.rglob("best.pt"):
            dest = WEIGHTS_DIR / "best.pt"
            pt_file.rename(dest)
            print(f"  ✓ Saved weights to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
            return True

        print("  ✗ No best.pt found in notebook output")
        return False

    except Exception as e:
        print(f"  ✗ Kaggle download failed: {e}")
        return False


# ───────────────────────────────────────────────────────────────────
# Method 2: Google Drive (MIC-DKFZ nnU-Net – 2nd place solution)
# ───────────────────────────────────────────────────────────────────

GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1uDLjtfIY0mDbwTPdvL0uWSRZHatJGjsS"


def try_gdrive():
    """Download the MIC-DKFZ model checkpoint from Google Drive."""
    print("\n[Method 2] Trying Google Drive (MIC-DKFZ nnU-Net checkpoint)...")

    try:
        import gdown
    except ImportError:
        print("  ✗ gdown not installed (pip install gdown)")
        return False

    out_dir = WEIGHTS_DIR / "_gdrive_ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        gdown.download_folder(GDRIVE_FOLDER, output=str(out_dir), quiet=False)
        files = list(out_dir.rglob("*"))
        if files:
            print(f"  ✓ Downloaded {len(files)} file(s) to {out_dir}")
            print("  Note: This is an nnU-Net checkpoint, not a YOLO .pt file.")
            print("  The webapp currently expects a YOLO .pt — see manual instructions below.")
            return True
        else:
            print("  ✗ Download completed but no files found (may be blocked by proxy)")
            return False
    except Exception as e:
        print(f"  ✗ Google Drive download failed: {e}")
        return False


# ───────────────────────────────────────────────────────────────────
# Method 3: Manual instructions
# ───────────────────────────────────────────────────────────────────

def print_manual():
    dest = WEIGHTS_DIR / "best.pt"
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║            Manual Download Instructions                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  OPTION A — Kaggle YOLO notebook (recommended, matches thesis)   ║
║                                                                  ║
║  1. Open this notebook in your browser:                          ║
║     https://www.kaggle.com/code/ravaghi/                         ║
║       flagellar-motor-detection-2-3-yolo-training                ║
║                                                                  ║
║  2. Click the "Output" tab at the top                            ║
║                                                                  ║
║  3. Download the best.pt file                                    ║
║     (usually under runs/detect/train/weights/best.pt)            ║
║                                                                  ║
║  4. Place it here:                                               ║
║     {dest}
║                                                                  ║
║  ────────────────────────────────────────────────────────────     ║
║                                                                  ║
║  OPTION B — Any Ultralytics YOLO .pt from the competition        ║
║                                                                  ║
║  1. Go to the competition code page:                             ║
║     https://www.kaggle.com/competitions/                         ║
║       byu-locating-bacterial-flagellar-motors-2025/code          ║
║                                                                  ║
║  2. Search for "YOLO" or "ultralytics"                           ║
║                                                                  ║
║  3. Open any notebook that trains YOLOv8                         ║
║                                                                  ║
║  4. Download best.pt from its Output tab                         ║
║                                                                  ║
║  5. Place it at: {dest}
║                                                                  ║
║  ────────────────────────────────────────────────────────────     ║
║                                                                  ║
║  OPTION C — Upload at runtime                                    ║
║                                                                  ║
║  Start the app (streamlit run app.py) and use the sidebar        ║
║  "Upload .pt" option to load weights interactively.              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="Download motor detection model weights")
    parser.add_argument("--kaggle", action="store_true", help="Only try Kaggle API")
    parser.add_argument("--gdrive", action="store_true", help="Only try Google Drive")
    parser.add_argument("--manual", action="store_true", help="Just print manual instructions")
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if (WEIGHTS_DIR / "best.pt").exists():
        size_mb = (WEIGHTS_DIR / "best.pt").stat().st_size / 1e6
        print(f"Weights already present: {WEIGHTS_DIR / 'best.pt'} ({size_mb:.1f} MB)")
        return

    if args.manual:
        print_manual()
        return

    print("=" * 60)
    print("Downloading model weights for Motor Detection webapp")
    print("=" * 60)

    if args.kaggle or not args.gdrive:
        if try_kaggle():
            return

    if args.gdrive or not args.kaggle:
        if try_gdrive():
            return

    print_manual()


if __name__ == "__main__":
    main()
