# src/eval.py
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str,
                           outpath: str = "reports/confusion_matrix.png"):
    """Matplotlib heatmap (no seaborn). Saves a nice PNG."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)  # default colormap works; we avoid specifying colors per your rules

    # ticks and labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    # write counts in cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("counts")

    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def eval_on_test(model, loader, class_names: List[str], device) -> Dict:
    """Prints report + confusion matrix; saves JSON + PNG."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb).cpu()
            y_true += yb.tolist()
            y_pred += out.argmax(1).tolist()

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    metrics = {
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, digits=4, output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # pretty heatmap
    _plot_confusion_matrix(cm, class_names, title="Recycle Classifier Confusion Matrix",
                           outpath="reports/confusion_matrix.png")

    print("Saved: reports/metrics.json and reports/confusion_matrix.png")
    return metrics
