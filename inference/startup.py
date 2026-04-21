"""
One-time startup tasks: ensure demo tomograms are available.

On Streamlit Cloud the repo won't include the ~600 MB of demo JPEG slices,
so we download them from HuggingFace the first time the app boots.

IMPORTANT: The .npy volumes on HuggingFace are ~1 GB each when loaded
fully into RAM.  Streamlit Cloud free tier only has 1 GB total, so we
use memory-mapped loading (mmap_mode='r') and process one slice at a
time to keep peak memory under ~100 MB.
"""

from __future__ import annotations

import gc
from pathlib import Path

import streamlit as st

DEMO_DIR = Path(__file__).resolve().parent.parent / "assets" / "demo_tomograms"

REPO_ID = "Floppanacci/tomogram-Bacterial-Flagellar-motors-location"

DEMO_TOMOS = [
    "tomo_00e047",
    "tomo_05df8a",
    "tomo_0a8f05",
    "tomo_0da370",
    "tomo_49f4ee",
]


def _tomo_ready(tomo_id: str) -> bool:
    d = DEMO_DIR / tomo_id
    return d.is_dir() and len(list(d.glob("*.jpg"))) > 5


def _download_one(tomo_id: str) -> bool:
    """
    Download a single .npy volume from HuggingFace and convert it to
    JPEG slices using memory-mapped I/O to stay within ~100 MB RAM.
    """
    import numpy as np
    from PIL import Image

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False

    out_dir = DEMO_DIR / tomo_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"train/{tomo_id}.npy",
            repo_type="dataset",
        )
    except Exception:
        return False

    vol = np.load(local, mmap_mode="r")
    n_slices = vol.shape[0]

    # Estimate percentiles from a sparse sample (~10 slices) instead of
    # loading the full volume into RAM for np.percentile().
    sample_indices = np.linspace(0, n_slices - 1, min(10, n_slices), dtype=int)
    sample_pixels = np.concatenate(
        [vol[i].ravel()[::64].astype(np.float32) for i in sample_indices]
    )
    p2 = float(np.percentile(sample_pixels, 2))
    p98 = float(np.percentile(sample_pixels, 98))
    del sample_pixels
    gc.collect()

    denom = max(p98 - p2, 1e-6)

    for z in range(n_slices):
        arr = vol[z].astype(np.float32)
        normed = np.clip((arr - p2) / denom, 0.0, 1.0)
        img = Image.fromarray((normed * 255).astype(np.uint8), mode="L")
        img.save(out_dir / f"slice_{z:04d}.jpg", quality=85)
        del arr, normed, img

    del vol
    gc.collect()
    return True


@st.cache_resource(show_spinner=False)
def ensure_demo_data() -> int:
    """
    Make sure all demo tomograms exist on disk.  Returns the count of
    tomograms that are ready.
    """
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    missing = [t for t in DEMO_TOMOS if not _tomo_ready(t)]
    if not missing:
        return len(DEMO_TOMOS)

    placeholder = st.empty()
    ok = len(DEMO_TOMOS) - len(missing)

    for i, tid in enumerate(missing, 1):
        placeholder.info(
            f"Downloading demo tomogram {i}/{len(missing)} "
            f"(**{tid}**) from HuggingFace — first run only…"
        )
        if _download_one(tid):
            ok += 1

    placeholder.empty()
    return ok
