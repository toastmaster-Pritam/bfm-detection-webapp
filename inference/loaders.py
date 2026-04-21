"""
Tomogram slice loaders for the Motor Detection webapp.

Three input modes:
  1. upload_images_to_slices  – list of in-memory UploadedFile objects (PNG/JPEG)
  2. mrc_to_slices            – single .mrc / .rec volume file
  3. demo_tomogram_slices     – read pre-downloaded demo data from disk
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Uploaded image files → ordered greyscale slices
# ---------------------------------------------------------------------------

def _natural_sort_key(name: str):
    """Sort filenames numerically so slice_9 < slice_10."""
    return [
        int(c) if c.isdigit() else c.lower()
        for c in re.split(r"(\d+)", name)
    ]


def upload_images_to_slices(
    uploaded_files,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Convert a list of Streamlit UploadedFile objects into ordered greyscale
    numpy arrays.

    Returns (slices, filenames) sorted by natural filename order.
    """
    pairs = []
    for uf in uploaded_files:
        img = Image.open(uf).convert("L")
        pairs.append((uf.name, np.asarray(img, dtype=np.uint8)))

    pairs.sort(key=lambda p: _natural_sort_key(p[0]))
    names = [p[0] for p in pairs]
    arrays = [p[1] for p in pairs]
    return arrays, names


# ---------------------------------------------------------------------------
# 2. MRC / REC volume → greyscale slices
# ---------------------------------------------------------------------------

def mrc_to_slices(file_bytes: bytes) -> list[np.ndarray]:
    """
    Decode a .mrc / .rec cryo-ET volume (provided as raw bytes) into a
    list of uint8 slices along axis 0 (z).
    """
    import mrcfile
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with mrcfile.open(tmp_path, permissive=True) as mrc:
            data = mrc.data.copy()
    finally:
        os.unlink(tmp_path)

    # Window to 8-bit using 2nd / 98th percentiles
    p2 = np.percentile(data, 2)
    p98 = np.percentile(data, 98)
    if p98 - p2 < 1e-6:
        norm = np.zeros_like(data, dtype=np.uint8)
    else:
        norm = np.clip((data - p2) / (p98 - p2), 0, 1)
        norm = (norm * 255).astype(np.uint8)

    return [norm[z] for z in range(norm.shape[0])]


# ---------------------------------------------------------------------------
# 3. Bundled demo tomograms
# ---------------------------------------------------------------------------

DEMO_DIR = Path(__file__).resolve().parent.parent / "assets" / "demo_tomograms"


def list_demo_tomograms() -> list[str]:
    """Return sorted folder names inside assets/demo_tomograms/."""
    if not DEMO_DIR.exists():
        return []
    return sorted(
        d.name for d in DEMO_DIR.iterdir()
        if d.is_dir() and (any(d.glob("*.jpg")) or any(d.glob("*.png")))
    )


def demo_tomogram_slices(name: str) -> list[np.ndarray]:
    """Load slices from a named demo tomogram folder."""
    tomo_dir = DEMO_DIR / name
    if not tomo_dir.is_dir():
        raise FileNotFoundError(f"Demo tomogram '{name}' not found in {DEMO_DIR}")

    files = sorted(
        [f for f in tomo_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda f: _natural_sort_key(f.name),
    )
    if not files:
        raise FileNotFoundError(f"No image files found in {tomo_dir}")

    return [np.asarray(Image.open(f).convert("L"), dtype=np.uint8) for f in files]
