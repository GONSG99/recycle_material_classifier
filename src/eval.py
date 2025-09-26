# src/eval.py
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("TkAgg")   # Force Tkinter backend (interactive window)
import matplotlib.pyplot as plt
from model import build_model
from data import load_data, build_transforms
import os
from PIL import Image

def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str,
                           outpath: str = "reports/confusion_matrix.png"):
    """Matplotlib heatmap (no seaborn). Saves a nice PNG."""
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

#from model import build_model
#from data import load_data
#import os

# ------------------- Show Raw Samples (No Transform) ------------------- (1 tab by itself)
from torchvision import datasets, transforms

def show_raw_samples(data_root, class_names, num_samples=5):
    """Display raw images without any transforms."""
    raw_dataset = datasets.ImageFolder(
        os.path.join(data_root, "train"), 
        transform=transforms.ToTensor()  # just convert to tensor for plotting
    )

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
    plt.suptitle("Raw Samples (No Transform)", fontsize=16)

    for i in range(num_samples):
        idx = np.random.randint(0, len(raw_dataset))
        image_tensor, label_idx = raw_dataset[idx]

        # PyTorch tensors are (C,H,W) → matplotlib wants (H,W,C)
        image = image_tensor.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        ax = axes[i]
        ax.imshow(image)
        ax.set_title(f"{class_names[label_idx]}")
        ax.axis("off")
    plt.show()


# ------------------- Show Transformed Samples -------------------(1 tab by itself)
def show_transformed_samples(dataset, class_names, num_samples=5):
    """Display images after transformations (what model sees)."""
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
    plt.suptitle("Transformed Samples (After train_tf/eval_tf)", fontsize=16)

    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        image_tensor, label_idx = dataset[idx]

        # Un-normalize for plotting
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        image = image_tensor.permute(1, 2, 0).numpy()
        image = std * image + mean
        image = np.clip(image, 0, 1)

        ax = axes[i]
        ax.imshow(image)
        ax.set_title(f"{class_names[label_idx]}")
        ax.axis("off")
    plt.show()

#------------------- Show RAW AND TRANSFORMED Samples -------------------(BOTH TOGETHER IN ONE tab :D)
def show_raw_and_transformed_samples(data_root: str, dataset, class_names: List[str], num_samples: int = 5):
    """Display raw images (top row) and transformed images (bottom row) in a single figure."""
    
    # Load raw images
    raw_dataset = datasets.ImageFolder(
        os.path.join(data_root, "train"), 
        transform=transforms.ToTensor()
    )

    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    plt.suptitle("Raw (Top) vs Transformed (Bottom) Samples", fontsize=16)

    for i in range(num_samples):
        # --- Raw image ---
        idx = np.random.randint(0, len(raw_dataset))
        raw_img, label_idx = raw_dataset[idx]
        raw_img = raw_img.permute(1, 2, 0).numpy()
        raw_img = np.clip(raw_img, 0, 1)
        axes[0, i].imshow(raw_img)
        axes[0, i].set_title(f"{class_names[label_idx]}")
        axes[0, i].axis("off")

        # --- Transformed image ---
        trans_img, label_idx = dataset[idx]  # use same idx for comparison
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        trans_img = trans_img.permute(1, 2, 0).numpy()
        trans_img = std * trans_img + mean
        trans_img = np.clip(trans_img, 0, 1)
        axes[1, i].imshow(trans_img)
        axes[1, i].set_title(f"{class_names[label_idx]}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

# ------------------- Main -------------------
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_ROOT = r"C:\Users\fearl\OneDrive\Desktop\recycle_material_classifier\data\images"
    MODEL_PATH = "models/resnet18_best.pt"
    BATCH_SIZE = 32

    print(f"Using device: {DEVICE}")
    print("Starting evaluation...")

    # Load transforms
    train_tf, eval_tf = build_transforms(img_size=224)

    # Load data
    data = load_data(DATA_ROOT, BATCH_SIZE)
    test_loader = data["loaders"]["test"]
    class_names = data["dsets"]["train"].classes
    print(f"Found {len(class_names)} classes: {class_names}")

    # Build and load model
    model = build_model(num_classes=len(class_names), freeze_backbone=False, device=DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Model weights loaded from {MODEL_PATH}")

    # Evaluate
    eval_on_test(model, test_loader, class_names, DEVICE)

    # Show raw + transformed samples (single window)
       # 1️⃣ Show raw images
    show_raw_samples(DATA_ROOT, class_names, num_samples=5)

    # 2️⃣ Show transformed images
    train_dataset = data["dsets"]["train"]
    show_transformed_samples(train_dataset, class_names, num_samples=5)

    train_dataset = data["dsets"]["train"]
    show_raw_and_transformed_samples(DATA_ROOT, train_dataset, class_names, num_samples=5)

    print("\nEvaluation complete.")

