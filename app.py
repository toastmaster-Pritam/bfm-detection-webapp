"""
Bacterial Flagellar Motor Detection — Streamlit Web Application

CNN--Transformer Ensemble (Config. E) + DBSCAN pipeline for automated
motor localisation in cryo-ET tomograms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BFM Detection — Cryo-ET",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS_PATH = Path(__file__).parent / "assets" / "styles.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

from inference.loaders import (
    demo_tomogram_slices,
    list_demo_tomograms,
    mrc_to_slices,
    upload_images_to_slices,
)
from inference.pipeline import detect_motors
from inference.startup import ensure_demo_data
from inference.visualize import build_3d_scatter, draw_boxes_on_slice

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEIGHTS_PATH = str(Path(__file__).parent / "weights" / "best.pt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# First-run: download demo tomograms if missing (cached, runs once)
# ---------------------------------------------------------------------------
ensure_demo_data()


# ═══════════════════════════════════════════════════════════════════════════
#  Sidebar — Pipeline Configuration
# ═══════════════════════════════════════════════════════════════════════════

def _sidebar() -> dict:
    st.sidebar.markdown("## Pipeline Configuration")

    st.sidebar.markdown("#### Detection")
    conf_thresh = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.05, max_value=0.95, value=0.25, step=0.05,
        help="Minimum per-box confidence score to keep a detection.",
    )

    st.sidebar.markdown("#### DBSCAN Clustering")
    dbscan_eps = st.sidebar.slider(
        "ε  (neighbourhood radius, voxels)",
        min_value=5, max_value=50, value=20, step=5,
        help="Spatial radius for DBSCAN grouping of slice-level detections.",
    )
    dbscan_min_samples = st.sidebar.slider(
        "min_samples",
        min_value=1, max_value=10, value=3, step=1,
        help="A motor must appear in at least this many slices to be confirmed.",
    )

    with st.sidebar.expander("About this pipeline"):
        st.markdown(
            "**Model:** CNN–Transformer Ensemble (Config. E)\n\n"
            "The ensemble fuses detections from a YOLOv8m backbone "
            "(CNN) and an RT-DETR-l backbone (Transformer) before "
            "applying density-based spatial clustering (DBSCAN) in "
            "3-D voxel space to consolidate slice-level predictions "
            "into motor centroids.\n\n"
            "**Post-processing:** DBSCAN with the optimal operating "
            "point (ε = 20, min_samples = 3) identified via grid "
            "search on the validation split.\n\n"
            "**Metric:** F₂-score with a 1 000 Å distance threshold, "
            "consistent with the MotorBench evaluation protocol.\n\n"
            f"**Device:** `{DEVICE}`"
        )

    return dict(
        conf_thresh=conf_thresh,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════════════

def _header():
    st.markdown(
        '<h1 style="margin-bottom:0;">Automated Detection of Bacterial '
        'Flagellar Motors in Cryo-ET</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">'
        'CNN–Transformer Ensemble + DBSCAN Pipeline &nbsp;·&nbsp; '
        'Configuration E &nbsp;·&nbsp; F<sub>2</sub> = 0.5974'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="header-rule">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Tab 1 — Input
# ═══════════════════════════════════════════════════════════════════════════

def _tab_input():
    mode = st.radio(
        "Choose input source",
        ["Demo tomogram", "Upload slice images", "Upload MRC volume"],
        horizontal=True,
    )

    slices: list[np.ndarray] | None = None
    tomo_label = ""

    if mode == "Demo tomogram":
        demos = list_demo_tomograms()
        if not demos:
            st.info(
                "No demo tomograms found. Run `python download_demo_data.py` "
                "to fetch sample data."
            )
        else:
            choice = st.selectbox(
                "Select a tomogram from the MotorBench validation set",
                demos,
                help="Each tomogram contains exactly one flagellar motor.",
            )
            slices = demo_tomogram_slices(choice)
            tomo_label = f"{choice}  ({len(slices)} slices)"
            h, w = slices[0].shape[:2]
            st.success(
                f"Loaded **{choice}** — {len(slices)} slices, "
                f"{w} × {h} px"
            )

            # Show a preview strip of evenly spaced slices
            n_preview = min(8, len(slices))
            indices = np.linspace(0, len(slices) - 1, n_preview, dtype=int)
            cols = st.columns(n_preview)
            for col, idx in zip(cols, indices):
                col.image(slices[idx], caption=f"z={idx}", use_container_width=True)

    elif mode == "Upload slice images":
        files = st.file_uploader(
            "Upload all PNG / JPEG slices from one tomogram",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        if files:
            slices, names = upload_images_to_slices(files)
            tomo_label = f"Uploaded ({len(slices)} slices)"
            st.success(
                f"Loaded **{len(slices)}** slices — "
                f"{slices[0].shape[1]} × {slices[0].shape[0]} px"
            )

    else:
        mrc_file = st.file_uploader(
            "Upload a .mrc or .rec tomogram file",
            type=["mrc", "rec"],
        )
        if mrc_file is not None:
            with st.spinner("Decoding MRC volume…"):
                slices = mrc_to_slices(mrc_file.getvalue())
            tomo_label = f"{mrc_file.name} ({len(slices)} slices)"
            st.success(
                f"Decoded **{len(slices)}** slices — "
                f"{slices[0].shape[1]} × {slices[0].shape[0]} px"
            )

    return slices, tomo_label


# ═══════════════════════════════════════════════════════════════════════════
#  Tab 2 — Slice Viewer
# ═══════════════════════════════════════════════════════════════════════════

def _tab_slice_viewer(slices, result):
    if slices is None or result is None:
        st.info("Load a tomogram and run detection first.")
        return

    n = len(slices)

    # Metrics row
    total_clustered = sum(1 for d in result.per_slice if d.cluster_id >= 0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total slices", n)
    c2.metric("Raw detections", result.n_raw)
    c3.metric("Clustered (confirmed)", total_clustered)
    c4.metric("Motors found", result.n_clustered)

    st.markdown("---")

    # Thumbnail mini-map
    n_thumbs = min(10, n)
    thumb_indices = np.linspace(0, n - 1, n_thumbs, dtype=int)
    thumb_cols = st.columns(n_thumbs)
    for col, idx in zip(thumb_cols, thumb_indices):
        det_count = sum(1 for d in result.per_slice if d.z == idx)
        border = "2px solid #48c78e" if det_count > 0 else "1px solid #444"
        col.markdown(
            f'<div style="border:{border}; border-radius:4px; '
            f'text-align:center; padding:2px;">'
            f'<span style="font-size:0.7rem; color:#aaa;">z={idx}</span></div>',
            unsafe_allow_html=True,
        )
        col.image(slices[idx], use_container_width=True)

    st.markdown("")

    # Z-slider
    z = st.slider("Navigate to slice", 0, n - 1, n // 2, key="z_slider")

    slice_dets = [d for d in result.per_slice if d.z == z]
    clustered = [d for d in slice_dets if d.cluster_id >= 0]

    col_img, col_info = st.columns([3, 1])

    with col_img:
        annotated = draw_boxes_on_slice(slices[z], result.per_slice, z)
        st.image(annotated, use_container_width=True)

    with col_info:
        st.markdown(f"**Slice z = {z}**")
        st.metric("Detections", len(slice_dets))
        st.metric("Confirmed motors", len(clustered))
        if clustered:
            for d in clustered:
                st.markdown(
                    f"<span style='color:#48c78e; font-weight:600;'>"
                    f"Motor {d.cluster_id}</span> — "
                    f"conf {d.confidence:.2f}",
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════
#  Tab 3 — 3-D Results
# ═══════════════════════════════════════════════════════════════════════════

def _tab_results_3d(result, tomo_label):
    if result is None:
        st.info("Load a tomogram and run detection first.")
        return

    n = result.n_clustered
    color = "#48c78e" if n > 0 else "#f14668"
    st.markdown(
        f'<div class="motor-count-banner" style="background:{color}15; '
        f'border:2px solid {color};">'
        f'<b>{n} motor{"s" if n != 1 else ""} detected</b>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'{result.n_raw} raw detections → DBSCAN → {n} cluster{"s" if n != 1 else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Full-width 3D scatter
    st.markdown("#### 3-D Motor Locations")
    fig = build_3d_scatter(result.motors_df)
    st.plotly_chart(fig, use_container_width=True)

    # Motor table
    if not result.motors_df.empty:
        st.markdown("#### Motor Coordinates")
        display_df = result.motors_df.copy()
        display_df.columns = ["Motor ID", "z", "y", "x",
                              "Mean Confidence", "# Detections", "z-Span"]
        for c in ["z", "y", "x", "z-Span"]:
            display_df[c] = display_df[c].round(1)
        display_df["Mean Confidence"] = display_df["Mean Confidence"].round(3)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Pipeline summary
    with st.expander("Pipeline summary", expanded=False):
        noise_count = result.n_raw - sum(
            1 for d in result.per_slice if d.cluster_id >= 0
        )
        st.markdown(
            f"- **Model:** CNN–Transformer Ensemble (Config. E)\n"
            f"- **Raw 2-D detections:** {result.n_raw}\n"
            f"- **DBSCAN clusters (motors):** {result.n_clustered}\n"
            f"- **Noise (suppressed singletons):** {noise_count}\n"
            f"- **Tomogram:** {tomo_label}"
        )

    # CSV download
    if not result.motors_df.empty:
        csv_data = result.motors_df[
            ["motor_id", "z", "y", "x", "mean_conf"]
        ].to_csv(index=False)
        st.download_button(
            "Download predictions CSV",
            data=csv_data,
            file_name="motor_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    _header()
    cfg = _sidebar()

    tab_input, tab_slice, tab_3d = st.tabs(
        ["📥  Input", "🔍  Slice Viewer", "📊  3-D Results"]
    )

    if "result" not in st.session_state:
        st.session_state.result = None
    if "slices" not in st.session_state:
        st.session_state.slices = None
    if "tomo_label" not in st.session_state:
        st.session_state.tomo_label = ""

    with tab_input:
        slices, tomo_label = _tab_input()
        if slices is not None:
            st.session_state.slices = slices
            st.session_state.tomo_label = tomo_label

        st.markdown("---")
        can_run = st.session_state.slices is not None

        if not can_run:
            st.warning("Select a demo tomogram or upload slices to begin.")

        run_clicked = st.button(
            "Run Ensemble Detection",
            disabled=not can_run,
            type="primary",
            use_container_width=True,
        )

        if run_clicked:
            progress_bar = st.progress(0, text="Running ensemble inference…")

            def _progress(current, total):
                progress_bar.progress(
                    current / total,
                    text=f"Processing slice {current} / {total}…",
                )

            with st.spinner("Running CNN–Transformer Ensemble + DBSCAN…"):
                result = detect_motors(
                    slices=st.session_state.slices,
                    weights_path=WEIGHTS_PATH,
                    conf_thresh=cfg["conf_thresh"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_samples=cfg["dbscan_min_samples"],
                    device=DEVICE,
                    progress_callback=_progress,
                )
                st.session_state.result = result

            progress_bar.progress(1.0, text="Done!")
            st.success(
                f"Detection complete — **{result.n_raw}** raw detections "
                f"→ **{result.n_clustered}** motor(s) after DBSCAN clustering."
            )

    with tab_slice:
        _tab_slice_viewer(st.session_state.slices, st.session_state.result)

    with tab_3d:
        _tab_results_3d(
            st.session_state.result, st.session_state.tomo_label
        )


if __name__ == "__main__":
    main()
