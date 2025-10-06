# -------------------------------------------------------------
# HOW TO RUN THIS SCRIPT:
# -------------------------------------------------------------
# 1️⃣ Create a virtual environment:
#     python -m venv .venv && .\.venv\Scripts\Activate
#
# 2️⃣ Install dependencies:
#     pip install -r requirements.txt
#
# 3️⃣ Train and evaluate model:
#     python src/main.py --data data/combined --epochs 15 --lr 3e-4 --bs 32
# -------------------------------------------------------------

import argparse, random
from pathlib import Path
from typing import Dict
import numpy as np
import torch

from data import load_data          
from model import build_model       
from train import train_model       
from eval import eval_on_test       

# -------------------------------------------------------------
# FUNCTION: set_seed
# -------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------------------------------------
# FUNCTION: make_dirs
# -------------------------------------------------------------
def make_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# FUNCTION: get_device
# -------------------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------
# FUNCTION: parse_args
# -------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Recycle Material Classifier (ResNet-18)")
    p.add_argument("--data", type=str, default="data/images", help="root with train/val/test")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--freeze", action="store_true", help="freeze ResNet backbone")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# -------------------------------------------------------------
# FUNCTION: main
# -------------------------------------------------------------
def main():

    # 1️⃣ Read input arguments
    args = parse_args()

    # 2️⃣ Ensure results are reproducible
    set_seed(args.seed)

    # 3️⃣ Create folders for saving models/reports
    make_dirs()

    # 4️⃣ Detect whether to use GPU or CPU
    device = get_device()
    print("Device:", device)

    # 5️⃣ Load dataset and dataloaders
    out = load_data(args.data, args.bs)
    dsets, loaders = out["dsets"], out["loaders"]
    # Extract class names in correct order
    class_names = [k for k, _ in sorted(dsets["train"].class_to_idx.items(), key=lambda x: x[1])]
    print("Classes:", class_names)

    # 6️⃣ Build model (ResNet18 with custom classification head)
    model = build_model(num_classes=len(class_names), freeze_backbone=args.freeze, device=device)

    # 7️⃣ Train model using training and validation sets
    model, best_val = train_model(model, loaders, epochs=args.epochs, lr=args.lr, device=device)
    print(f"Best val acc: {best_val:.4f}")

    # 8️⃣ Evaluate final model on the test set and save reports
    eval_on_test(model, loaders["test"], class_names, device)

# -------------------------------------------------------------
# ENTRY POINT (runs when executed directly)
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
