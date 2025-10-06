# -------------------------------------------------------------
# This script handles ALL dataset-related logic:
# 1. Builds image transformations for training and testing
# 2. Loads datasets from "train", "val", "test" folders
# 3. Creates DataLoaders for batching
# 4. Exports label names to "models/labels.json" for later use
# -------------------------------------------------------------

from pathlib import Path
import os, json, torch
from typing import Dict, Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ------------------------------
# BUILD TRANSFORMS (AUGMENTATION)
# ------------------------------
def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:

    # --- Training transforms ---
    # These help the model generalize better by creating variations of input images
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(45, fill=(0,)),  # Background filled with black
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # --- Evaluation transforms ---
    # Only resize + normalize (no randomness to keep results consistent)
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


# ------------------------------
# LOAD DATASETS & BUILD LOADERS
# ------------------------------
# Load datasets and create data loaders
def load_data(data_root: str, bs: int, num_workers: int = 2) -> Dict:

    # Get transformation pipelines
    train_tf, eval_tf = build_transforms(224)

    # --- Load image folders ---
    # Each folder should look like:
    # data_root/train/class_name/*.jpg
    # data_root/val/class_name/*.jpg
    # data_root/test/class_name/*.jpg
    dsets = {
        "train": datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf),
        "val":   datasets.ImageFolder(os.path.join(data_root, "val"),   transform=eval_tf),
        "test":  datasets.ImageFolder(os.path.join(data_root, "test"),  transform=eval_tf),
    }

    # --- Save label mapping to file ---
    # This creates a file: models/labels.json
    # Example content: {"0": "paper", "1": "plastic", "2": "metal"}
    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/labels.json", "w") as f:
        json.dump({v: k for k, v in dsets["train"].class_to_idx.items()}, f, indent=2)

    # --- Check if GPU is available ---
    # If yes, use "pinned memory" to load faster
    pin = torch.cuda.is_available()

    # --- Build DataLoaders ---
    # DataLoader helps in batching, shuffling, and parallel loading
    loaders = {
        "train": DataLoader(dsets["train"], batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=pin),
        "val":   DataLoader(dsets["val"],   batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin),
        "test":  DataLoader(dsets["test"],  batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin),
    }

    # Return both datasets and loaders for training scripts
    return {"dsets": dsets, "loaders": loaders}
