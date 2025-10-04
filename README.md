# ♻️ Recycle Material Classifier (ResNet‑18)

> Simple transfer‑learning project for AAI3001: classify **paper / plastic / metal / others** from images, train a small model, and serve a **Gradio** demo with **Grad‑CAM** heatmaps.

---

## TL;DR

* **Model:** ResNet‑18 pre‑trained on ImageNet; last layer swapped to 4 classes; optional **freeze** of the backbone.
* **Data:** Kaggle trash‑type dataset → keep *paper/plastic/metal*, merge *glass/cardboard/trash → others*; can mix in your **own photos**.
* **Training:** AdamW + ReduceLROnPlateau; early‑stop style via patience; TensorBoard logging.
* **Outputs:** `models/resnet18_best.pt`, `models/labels.json`, `reports/curves_final.png`, `reports/confusion_matrix.png`.
* **App:** `src/app.py` Gradio UI (upload image + live IP‑webcam) with **Grad‑CAM** overlays.

---

## Repo structure

```
recycle_material_classifier/
├─ data/                         # datasets live here (created by scripts)
│  ├─ images/                    # Kaggle, flattened into train/val/test
│  ├─ own_images/                # your photos (split by script)
│  └─ combined/                  # Kaggle + own merged set
├─ models/
│  ├─ resnet18_best.pt          # best checkpoint (auto‑saved)
│  └─ labels.json               # class index ↔ name (auto‑saved)
├─ reports/
│  ├─ curves_final.png          # training curves
│  └─ confusion_matrix.png      # test confusion matrix
├─ runs/recycle/                 # TensorBoard logs
└─ src/
   ├─ app.py                    # Gradio app + Grad‑CAM + live webcam
   ├─ main.py                   # CLI pipeline (data→train→eval)
   ├─ model.py                  # ResNet‑18 factory (freeze option)
   ├─ data.py                   # transforms + DataLoaders + labels.json
   ├─ train.py                  # training loop, early stop, plots
   ├─ eval.py                   # metrics + confusion matrix
   ├─ dataset.py                # download/flatten Kaggle dataset
   ├─ own_images.py             # split your photos train/val/test
   └─ images_combined.py        # merge Kaggle + own into /data/combined
```

---

## Quickstart

### 1) Setup

**Windows (PowerShell)**

```powershell
cd "path\to\recycle_material_classifier"
python -m venv .venv
. .venv\Scripts\Activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
cd /path/to/recycle_material_classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare data

Option A — **Kaggle set only** (auto‑download & split):

```bash
python src/dataset.py            # creates data/images/train|val|test/(paper|plastic|metal|others)
```

Option B — **Add your own photos** (put raw images under `data/own_images/<class>/` first):

```bash
python src/own_images.py         # splits your photos into train/val/test
python src/images_combined.py    # merges Kaggle + own → data/combined
```

### 3) Train

```bash
# Use Kaggle‑only set
python src/main.py --data data/images --epochs 15 --lr 3e-4 --bs 32

# Or the merged set
python src/main.py --data data/combined --epochs 15 --lr 3e-4 --bs 32

# (Optional) freeze the backbone
python src/main.py --data data/combined --freeze
```

Artifacts will appear in `models/` and `reports/`.

### 4) Evaluate (auto‑runs after training)

* Saves overall accuracy, classification report JSON, and **confusion matrix** plot to `reports/`.

### 5) Run the demo app

Run **from the project root** (not inside `src/`):

```bash
python src/app.py
```

* Upload an image to see **prediction + confidence** and **Grad‑CAM** heatmaps.
* (Optional) In the *Live IP Webcam* tab, adjust `ip_url` in `src/app.py` to your phone’s IP Webcam feed.

---

## Model details (short)

* **Backbone:** ResNet‑18 pre‑trained on ImageNet.
* **Head:** Replace the final `fc` with a new `Linear(in_features, 4)`.
* **Freeze option:** `--freeze` locks backbone weights; otherwise we **fine‑tune** end‑to‑end.
* **Input:** `224×224` RGB, **ImageNet** mean/std normalization.

## Training recipe

* **Optimizer:** AdamW (weight decay `1e-4`).
* **Scheduler:** ReduceLROnPlateau on **val accuracy**.
* **Early stop style:** stop after no val‑acc improvement for `patience=3` epochs.
* **Logging:** TensorBoard under `runs/recycle/` + saved **curves** image.
* **Reproducibility:** `seed=42`, deterministic flags set.

## Evaluation

* Prints a **classification report**, computes **confusion matrix** & **accuracy**.
* Saves `reports/metrics.json` + `reports/confusion_matrix.png`.

## Explainability (Grad‑CAM)

* The app generates an **overlay** and a **heatmap** for the predicted class.
* Default target layer: `layer4` block of ResNet‑18.

---

## Tips & Troubleshooting

* **Run from root:** If you see `FileNotFoundError: models/labels.json`, you likely executed inside `src/`. Use `python src/app.py` from the project root.
* **ModuleNotFoundError for `src.app`:** The repo isn’t a Python package; run the file path variant above.
* **No GPU?** The code auto‑falls back to CPU.
* **Webcam feed shows “offline”:** Check `ip_url` and that your phone and laptop share the same network.
* **Class names wrong/missing:** Ensure `models/labels.json` exists (it’s written during `load_data` at train time).

---

## Acknowledgements

* Pre‑trained **ResNet‑18** from `torchvision.models`.
* Trash‑type dataset by the original Kaggle author (used here for course learning).

---

## License

For learning and coursework purposes (AAI3001). If you plan to release publicly, add an explicit license.
