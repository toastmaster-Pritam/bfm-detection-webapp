# Bacterial Flagellar Motor Detection — Streamlit Webapp

Interactive web application for detecting bacterial flagellar motors in cryo-ET
tomograms using the **CNN–Transformer Ensemble (Config. E) + DBSCAN** pipeline.

## Quick start (local)

```bash
cd webapp
pip install -r requirements.txt
streamlit run app.py
```

The app opens at **http://localhost:8501**.

On first launch the app auto-downloads 5 demo tomograms from HuggingFace
(~75 MB total, cached afterwards).

## Features

| Feature | Description |
|---------|-------------|
| **Demo tomograms** | 5 real MotorBench tomograms, auto-downloaded on first run |
| **Upload slices** | Drag-and-drop PNG / JPEG slice images from a tomogram folder |
| **Upload MRC** | Load a `.mrc` / `.rec` volume directly (decoded with `mrcfile`) |
| **Slice viewer** | Scrub through z-slices with bounding-box overlays and mini-map |
| **3-D scatter** | Interactive Plotly 3-D view of clustered motor centres |
| **CSV export** | Download motor predictions as CSV |

## Deploy to Streamlit Community Cloud

### 1. Push to GitHub

```bash
cd webapp
git init
git add .
git commit -m "Initial commit — BFM Detection webapp"
```

Create a repo on GitHub (e.g. `bfm-detection-webapp`), then:

```bash
git remote add origin https://github.com/<your-username>/bfm-detection-webapp.git
git branch -M main
git push -u origin main
```

> **Note:** `weights/best.pt` (18 MB) is committed to the repo.  
> Demo tomograms (~75 MB) are **not** committed — they are auto-downloaded
> from HuggingFace on the first cloud run.

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **New app**
4. Select your repo, branch `main`, main file `app.py`
5. Click **Deploy**

Streamlit Cloud will:
- Install everything in `requirements.txt`
- Apply the theme from `.streamlit/config.toml`
- Auto-download demo tomograms on first access

Your app will be live at `https://<app-name>.streamlit.app`.

### Resource notes

- **Free tier:** 1 GB RAM, CPU only. Inference on 300 slices takes ~2–5 min.
- **Weights:** `best.pt` is included in the repo (18 MB).
- **Demo data:** Downloaded once per container start via `huggingface_hub`.
- **Secrets:** If you need a HuggingFace token, add it in Streamlit Cloud
  Settings → Secrets as `HF_TOKEN = "hf_..."`.

## Directory layout

```
webapp/
├── app.py                  # Streamlit entry point
├── inference/
│   ├── pipeline.py         # 2.5D encoding → YOLO → DBSCAN
│   ├── loaders.py          # Slice / MRC / demo loaders
│   ├── visualize.py        # Bbox overlay + Plotly 3-D scatter
│   └── startup.py          # Auto-download demo data on first run
├── assets/
│   ├── demo_tomograms/     # Auto-populated from HuggingFace
│   └── styles.css          # Custom CSS theme
├── weights/
│   └── best.pt             # YOLOv8 checkpoint (18 MB)
├── .streamlit/
│   └── config.toml         # Server + theme config
├── download_demo_data.py   # Manual demo download script
├── download_weights.py     # Manual weights download script
├── requirements.txt
├── .gitignore
└── README.md
```