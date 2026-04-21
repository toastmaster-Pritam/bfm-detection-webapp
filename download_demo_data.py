#!/usr/bin/env python3
"""
Download demo tomograms from the BYU MotorBench dataset on HuggingFace
and save them to assets/demo_tomograms/ for use with the Streamlit webapp.

Usage:
    cd webapp
    python download_demo_data.py

Each demo tomogram is stored as a folder of JPEG slice images.

Source dataset:
  https://huggingface.co/datasets/Floppanacci/tomogram-Bacterial-Flagellar-motors-location
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DEMO_DIR = Path(__file__).resolve().parent / "assets" / "demo_tomograms"

REPO_ID = "Floppanacci/tomogram-Bacterial-Flagellar-motors-location"

DEMO_TOMOS = [
    "tomo_00e047",
    "tomo_05df8a",
    "tomo_0a8f05",
    "tomo_0da370",
    "tomo_49f4ee",
]


def _normalize(vol: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(vol, 2), np.percentile(vol, 98)
    if p98 - p2 < 1e-6:
        return np.zeros_like(vol, dtype=np.uint8)
    return np.uint8(255.0 * np.clip((vol - p2) / (p98 - p2), 0, 1))


def _download_one(tomo_id: str) -> bool:
    from huggingface_hub import hf_hub_download

    out_dir = DEMO_DIR / tomo_id
    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) > 5:
        print(f"  {tomo_id} already present ({len(list(out_dir.glob('*.jpg')))} slices), skipping.")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    npy_file = f"train/{tomo_id}.npy"
    print(f"  Downloading {npy_file} ...")
    try:
        local = hf_hub_download(repo_id=REPO_ID, filename=npy_file, repo_type="dataset")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    vol = np.load(local)
    print(f"  Volume shape: {vol.shape}, converting to slices ...")
    norm = _normalize(vol)
    for z in range(norm.shape[0]):
        Image.fromarray(norm[z], mode="L").save(out_dir / f"slice_{z:04d}.jpg", quality=90)
    print(f"  Saved {norm.shape[0]} slices to {out_dir}")
    return True


def main():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading demo tomograms for the BFM Detection webapp")
    print("=" * 60)

    ok = 0
    for tomo_id in DEMO_TOMOS:
        print(f"\n--- {tomo_id} ---")
        if _download_one(tomo_id):
            ok += 1

    print(f"\nDownloaded {ok}/{len(DEMO_TOMOS)} tomograms.")

    print("\nCurrent demos:")
    for d in sorted(DEMO_DIR.iterdir()):
        if d.is_dir():
            print(f"  {d.name}: {len(list(d.glob('*.jpg')))} slices")

    print("\nDone! Run: streamlit run app.py --server.headless true")


if __name__ == "__main__":
    main()
