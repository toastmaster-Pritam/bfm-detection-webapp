# Model Weights

This folder should contain a **YOLOv8 `.pt` checkpoint** trained on the
BYU MotorBench cryo-ET dataset.  The app looks for `best.pt` here by default.

## Quickest way to get weights

### Step 1: Open this Kaggle notebook

<https://www.kaggle.com/code/ravaghi/flagellar-motor-detection-2-3-yolo-training>

### Step 2: Click the **Output** tab (top of the page)

The notebook trains a YOLOv8 model and saves `best.pt` in its output.

### Step 3: Download `best.pt`

It's typically at `runs/detect/train/weights/best.pt` inside the output.

### Step 4: Place it here

```
webapp/weights/best.pt
```

That's it.  Start the app with `streamlit run app.py`.

---

## Alternative methods

### Use the download helper script

```bash
cd webapp
python download_weights.py
```

This script tries multiple download methods (Kaggle API, Google Drive)
and prints detailed manual instructions as a fallback.

### Upload at runtime

Don't want to place files on disk?  Start the app and switch the sidebar
to **"Upload .pt"** — you can drag-and-drop your weights file directly.

### Train your own

Follow the procedure in `notebooks/03_training_yolov8_rtdetr.ipynb` from
this repository, then copy the resulting `best.pt` here.

### Any other YOLO checkpoint from the competition

Browse the competition code page:

<https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/code>

Search for "YOLO" or "ultralytics".  Any public notebook that trains
YOLOv8 on this data will produce a compatible `best.pt`.

## Supported architectures

The inference pipeline uses `ultralytics.YOLO(path)`, which auto-detects
the model variant (YOLOv8n/s/m/l/x, YOLOv9, YOLOv10, YOLO11, RT-DETR).
Any `.pt` trained via the Ultralytics framework on the MotorBench data
should work without code changes.

## File checklist

```
webapp/weights/
├── best.pt          ← your trained checkpoint (not tracked by git)
└── README.md        ← this file
```
