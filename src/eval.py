import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Plot confusion matrix as a heatmap
def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, outpath: str = "reports/confusion_matrix.png"):

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("counts")

    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

# Evaluate model on test set and print/save metrics
def eval_on_test(model, loader, class_names: List[str], device) -> Dict:

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

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:\n", cm)
    
    total = cm.sum()
    acc = float(cm.trace()) / float(total) if total > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "labels": class_names,
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, digits=4, output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Pretty heatmap
    _plot_confusion_matrix(
        cm,
        class_names,
        title="Recycle Classifier Confusion Matrix",
        outpath="reports/confusion_matrix.png",
    )

    print("Saved: reports/metrics.json and reports/confusion_matrix.png")
    return metrics

