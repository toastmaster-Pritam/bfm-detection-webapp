"""
One-time startup tasks: ensure demo tomograms are available.

On Streamlit Cloud the repo won't include the ~75 MB of demo JPEG slices,
so we download them from HuggingFace the first time the app boots.
The result is cached with @st.cache_resource so it runs only once per
deployment / container restart.
"""

from __future__ import annotations

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
    """Download a single .npy volume and convert to JPEG slices."""
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

    vol = np.load(local)
    p2, p98 = np.percentile(vol, 2), np.percentile(vol, 98)
    if p98 - p2 < 1e-6:
        norm = np.zeros_like(vol, dtype=np.uint8)
    else:
        norm = np.uint8(255.0 * np.clip((vol - p2) / (p98 - p2), 0, 1))

    for z in range(norm.shape[0]):
        Image.fromarray(norm[z], mode="L").save(
            out_dir / f"slice_{z:04d}.jpg", quality=90
        )
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
    placeholder.info(
        f"Downloading {len(missing)} demo tomogram(s) from HuggingFace "
        f"(first run only)…"
    )

    ready = len(DEMO_TOMOS) - len(missing)
    for tid in missing:
        if _download_one(tid):
            ready += 1

    placeholder.empty()
    return ready
