# -------------------------------------------------------------
# This script evaluates your trained model:
# Runs inference on the test dataset
# Calculates accuracy, precision, recall, F1-score
# Plots a confusion matrix
# Saves everything in the /reports folder
# -------------------------------------------------------------

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# HELPER FUNCTION: Plot Confusion Matrix
# -------------------------------------------------------------
def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, outpath: str = "reports/confusion_matrix.png"):

    cm = np.array(cm)  # Ensure it’s a NumPy array

     # Create figure and axes for plotting
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm) # Display matrix as an image (heatmap)

    # Label the x and y axes with class names
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    # Add numeric values inside each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    # Add colorbar to show intensity scale
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("counts") # Label for the colorbar

    fig.tight_layout()

    # Save the figure to file
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

# -------------------------------------------------------------
# MAIN FUNCTION: Evaluate Model on Test Data
# -------------------------------------------------------------
def eval_on_test(model, loader, class_names: List[str], device) -> Dict:

    model.eval() # Switch model to evaluation mode (disables dropout/batchnorm)
    y_true, y_pred = [], [] # Store true and predicted labels

    # Disable gradient computation (faster inference)
    with torch.no_grad():
        for xb, yb in loader:                       # Loop through test batches     
            xb = xb.to(device, non_blocking=True)   # Move data to device
            out = model(xb).cpu()                   # Forward pass and move output to CPU 
            y_true += yb.tolist()                   # Collect true labels
            y_pred += out.argmax(1).tolist()        # Collect predicted labels

    # ---- PRINT TEXT REPORT ----
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # ---- CONFUSION MATRIX ----
    labels = list(range(len(class_names))) # Numeric labels [0, 1, 2, ...]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:\n", cm)
    
    # ---- CALCULATE OVERALL ACCURACY ----
    total = cm.sum()
    acc = float(cm.trace()) / float(total) if total > 0 else 0.0 # Correct predictions / Total predictions

    # ---- SAVE METRICS INTO A DICTIONARY ----
    metrics = {
        "accuracy": acc,
        "labels": class_names,
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, digits=4, output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
    }

    # ---- SAVE METRICS TO FILE ----
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- PLOT AND SAVE CONFUSION MATRIX IMAGE ----
    _plot_confusion_matrix(
        cm,
        class_names,
        title="Recycle Classifier Confusion Matrix",
        outpath="reports/confusion_matrix.png",
    )

    print("Saved: reports/metrics.json and reports/confusion_matrix.png")
    return metrics

