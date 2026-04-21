"""
2.5D YOLOv8 + DBSCAN motor-detection pipeline.

Adapted from notebooks/04_inference_dbscan.ipynb.  All Kaggle-specific paths
and notebook state have been removed; the public API is a single function
`detect_motors()` that accepts a list of greyscale slices and returns
per-slice detections plus a DataFrame of DBSCAN-clustered motor centres.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class SliceDetection:
    z: int
    y: float
    x: float
    y1: float
    x1: float
    y2: float
    x2: float
    confidence: float
    cluster_id: int = -1  # -1 = noise / un-clustered


@dataclass
class DetectionResult:
    per_slice: list[SliceDetection] = field(default_factory=list)
    motors_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    n_raw: int = 0
    n_clustered: int = 0


# ---------------------------------------------------------------------------
# 2.5D encoding helpers
# ---------------------------------------------------------------------------

def _normalize_slice(arr: np.ndarray) -> np.ndarray:
    """Percentile-based contrast stretch to uint8."""
    p2 = np.percentile(arr, 2)
    p98 = np.percentile(arr, 98)
    if p98 - p2 < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    clipped = np.clip(arr, p2, p98)
    return np.uint8(255.0 * (clipped - p2) / (p98 - p2))


def _build_rgb(slices: list[np.ndarray], z: int) -> np.ndarray:
    """Stack (z-1, z, z+1) as an RGB image for the 2.5D encoding."""
    z_max = len(slices)
    channels = []
    for dz in (-1, 0, 1):
        idx = max(0, min(z + dz, z_max - 1))
        channels.append(_normalize_slice(slices[idx].astype(np.float32)))
    return np.stack(channels, axis=-1)


# ---------------------------------------------------------------------------
# DBSCAN clustering
# ---------------------------------------------------------------------------

def _dbscan_cluster(
    detections: list[SliceDetection],
    eps: float,
    min_samples: int,
) -> tuple[list[SliceDetection], pd.DataFrame]:
    """
    Cluster raw 2-D detections in 3-D (z, y, x) space.

    Returns the detections (with `cluster_id` assigned) and a DataFrame of
    motor centres with columns:
      motor_id | z | y | x | mean_conf | n_detections | z_span
    """
    if len(detections) < min_samples:
        return detections, pd.DataFrame(
            columns=["motor_id", "z", "y", "x",
                     "mean_conf", "n_detections", "z_span"])

    coords = np.array([[d.z, d.y, d.x] for d in detections])
    confs = np.array([d.confidence for d in detections])

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)

    for det, lbl in zip(detections, labels):
        det.cluster_id = int(lbl)

    unique = sorted(set(labels) - {-1})
    rows = []
    for cid in unique:
        mask = labels == cid
        cl_coords = coords[mask]
        cl_confs = confs[mask]
        weights = cl_confs / cl_confs.sum()
        centroid = (cl_coords * weights[:, None]).sum(axis=0)
        rows.append({
            "motor_id": cid,
            "z": float(centroid[0]),
            "y": float(centroid[1]),
            "x": float(centroid[2]),
            "mean_conf": float(cl_confs.mean()),
            "n_detections": int(mask.sum()),
            "z_span": float(cl_coords[:, 0].max() - cl_coords[:, 0].min()),
        })

    motors_df = pd.DataFrame(rows)
    return detections, motors_df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_motors(
    slices: list[np.ndarray],
    weights_path: str,
    conf_thresh: float = 0.25,
    dbscan_eps: float = 20.0,
    dbscan_min_samples: int = 3,
    device: str = "cpu",
    batch_size: int = 8,
    progress_callback=None,
) -> DetectionResult:
    """
    End-to-end motor detection on a stack of greyscale slices.

    Parameters
    ----------
    slices : list of 2-D numpy arrays (H x W), uint8 or float.
    weights_path : path to a YOLOv8 `best.pt` checkpoint.
    conf_thresh : YOLO confidence threshold.
    dbscan_eps : DBSCAN neighbourhood radius in voxel units.
    dbscan_min_samples : minimum detections per cluster.
    device : "cpu" or "cuda:0".
    batch_size : slices per YOLO forward pass.
    progress_callback : optional callable(current, total) for progress bars.

    Returns
    -------
    DetectionResult with per-slice detections and clustered motors DataFrame.
    """
    from ultralytics import YOLO  # deferred so import doesn't block startup

    model = YOLO(weights_path)
    model.to(device)

    n_slices = len(slices)
    all_dets: list[SliceDetection] = []

    with tempfile.TemporaryDirectory(prefix="motor_webapp_") as tmp:
        for batch_start in range(0, n_slices, batch_size):
            batch_end = min(batch_start + batch_size, n_slices)
            batch_paths: list[str] = []
            batch_z: list[int] = []

            for z in range(batch_start, batch_end):
                rgb = _build_rgb(slices, z)
                path = os.path.join(tmp, f"z{z:04d}.jpg")
                Image.fromarray(rgb).save(path, quality=90)
                batch_paths.append(path)
                batch_z.append(z)

            if not batch_paths:
                continue

            results = model(batch_paths, verbose=False, conf=conf_thresh)

            for j, result in enumerate(results):
                if result.boxes is not None and len(result.boxes) > 0:
                    for bi in range(len(result.boxes)):
                        x1, y1, x2, y2 = (
                            result.boxes.xyxy[bi].cpu().numpy()
                        )
                        all_dets.append(SliceDetection(
                            z=batch_z[j],
                            y=float((y1 + y2) / 2),
                            x=float((x1 + x2) / 2),
                            y1=float(y1), x1=float(x1),
                            y2=float(y2), x2=float(x2),
                            confidence=float(result.boxes.conf[bi]),
                        ))

            if progress_callback:
                progress_callback(batch_end, n_slices)

    clustered_dets, motors_df = _dbscan_cluster(
        all_dets, eps=dbscan_eps, min_samples=dbscan_min_samples,
    )

    return DetectionResult(
        per_slice=clustered_dets,
        motors_df=motors_df,
        n_raw=len(all_dets),
        n_clustered=len(motors_df),
    )
