"""
Visualization helpers for the Motor Detection webapp.

  draw_boxes_on_slice  – overlay bounding boxes on a single z-slice image.
  build_3d_scatter     – interactive Plotly 3-D scatter of clustered motor centres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    import pandas as pd
    from .pipeline import SliceDetection


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
CLUSTERED_COLOR = (72, 199, 142)      # green – belongs to a valid cluster
NOISE_COLOR = (158, 158, 158)          # grey – singleton / noise
FP_ALPHA = 100                         # transparency for noise boxes

_FONT = None


def _font(size: int = 12):
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            try:
                _FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            except OSError:
                _FONT = ImageFont.load_default()
    return _FONT


# ---------------------------------------------------------------------------
# Bounding-box overlay on a single slice
# ---------------------------------------------------------------------------

def draw_boxes_on_slice(
    img_array: np.ndarray,
    detections: list["SliceDetection"],
    z: int,
) -> Image.Image:
    """
    Draw bounding boxes for slice *z* on top of the greyscale image.

    Clustered detections (cluster_id >= 0) are drawn green;
    noise detections (cluster_id == -1) are drawn grey with lower opacity.

    Returns a PIL RGBA image.
    """
    if img_array.ndim == 2:
        base = Image.fromarray(img_array, mode="L").convert("RGBA")
    else:
        base = Image.fromarray(img_array).convert("RGBA")

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font()

    slice_dets = [d for d in detections if d.z == z]

    # Draw noise first (underneath), then clustered on top
    for d in sorted(slice_dets, key=lambda d: (d.cluster_id >= 0, d.confidence)):
        if d.cluster_id >= 0:
            color = CLUSTERED_COLOR + (220,)
            lw = 3
            label = f"motor {d.cluster_id}  {d.confidence:.2f}"
        else:
            color = NOISE_COLOR + (FP_ALPHA,)
            lw = 1
            label = f"{d.confidence:.2f}"

        box = [d.x1, d.y1, d.x2, d.y2]
        for offset in range(lw):
            draw.rectangle(
                [box[0] - offset, box[1] - offset,
                 box[2] + offset, box[3] + offset],
                outline=color,
            )

        # Label background
        tw = len(label) * 7
        th = 14
        lx, ly = d.x1, max(d.y1 - th - 2, 0)
        draw.rectangle([lx, ly, lx + tw, ly + th], fill=color)
        draw.text((lx + 2, ly + 1), label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(base, overlay)


# ---------------------------------------------------------------------------
# 3-D scatter of motor centres (Plotly)
# ---------------------------------------------------------------------------

def build_3d_scatter(motors_df: "pd.DataFrame"):
    """
    Create a Plotly Figure showing clustered motor centres in 3-D.

    Expects a DataFrame with columns: motor_id, z, y, x, mean_conf, n_detections.
    """
    import plotly.graph_objects as go

    if motors_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No motors detected",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=20, color="grey"),
        )
        fig.update_layout(height=500)
        return fig

    fig = go.Figure(data=[
        go.Scatter3d(
            x=motors_df["x"],
            y=motors_df["y"],
            z=motors_df["z"],
            mode="markers+text",
            marker=dict(
                size=motors_df["n_detections"].clip(upper=20) * 1.5 + 5,
                color=motors_df["mean_conf"],
                colorscale="Viridis",
                cmin=0.0,
                cmax=1.0,
                colorbar=dict(title="Mean conf."),
                opacity=0.88,
                line=dict(width=1, color="white"),
            ),
            text=[f"Motor {row.motor_id}" for _, row in motors_df.iterrows()],
            hovertemplate=(
                "<b>Motor %{text}</b><br>"
                "x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}<br>"
                "conf: %{marker.color:.3f}<extra></extra>"
            ),
        )
    ])

    fig.update_layout(
        scene=dict(
            xaxis_title="x (voxels)",
            yaxis_title="y (voxels)",
            zaxis_title="z (slice)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=550,
    )
    return fig
