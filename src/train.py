# src/train.py
import json, time
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from typing import Tuple, Dict, List
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


# ---------- small utilities ----------
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

def _fmt_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    return f"{m:02d}:{s:02d}"

def _get_targets(ds) -> torch.Tensor:
    if hasattr(ds, "targets"):
        return torch.as_tensor(ds.targets)
    # ImageFolder fallback
    return torch.as_tensor([t for _, t in ds.samples])

def _class_counts(ds) -> List[str]:
    if not hasattr(ds, "classes"): return []
    y = _get_targets(ds)
    counts = torch.bincount(y, minlength=len(ds.classes)).tolist()
    return [f"{cls}({cnt})" for cls, cnt in zip(ds.classes, counts)]

def _num_params(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

def _current_lrs(optimizer) -> str:
    lrs = sorted({pg["lr"] for pg in optimizer.param_groups})
    if len(lrs) == 1: return f"{lrs[0]:.2e}"
    return ", ".join(f"{lr:.2e}" for lr in lrs)


# ---------- one epoch (with a live progress bar) ----------
def run_epoch(model,loader,criterion,optimizer,device,train: bool,epoch: int,epochs: int,) -> Tuple[float, float]:
    model.train() if train else model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    total = 0

    bar = tqdm(
        loader,
        desc=f"[{epoch:02d}/{epochs}] {'train' if train else 'val  '}",
        leave=False,
        dynamic_ncols=True,
    )

    for xb, yb in bar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                optimizer.step()

        bsz = xb.size(0)
        epoch_loss += loss.item() * bsz
        epoch_acc  += accuracy_from_logits(out, yb) * bsz
        total      += bsz

        # running averages in the bar
        bar.set_postfix(loss=f"{epoch_loss/max(total,1):.4f}",
                        acc=f"{epoch_acc/max(total,1):.4f}")

    return epoch_loss / total, epoch_acc / total


# ---------- curves / history ----------
def _save_curve(history, epoch, best_val_acc, outdir="reports"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    xs = list(range(1, epoch + 1))

    plt.figure(figsize=(7, 5))
    plt.plot(xs, history["train_acc"][:epoch], label="train_acc")
    plt.plot(xs, history["val_acc"][:epoch],   label="val_acc")
    plt.plot(xs, history["train_loss"][:epoch], label="train_loss")
    plt.plot(xs, history["val_loss"][:epoch],   label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("value")
    plt.title(f"Learning Curves (up to epoch {epoch})  |  best val acc = {best_val_acc:.4f}")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{outdir}/curve_ep{epoch:02d}.png", dpi=200)
    plt.close()

    with open(f"{outdir}/history.json", "w") as f:
        json.dump(history, f, indent=2)


# ---------- main training loop ----------
def train_model(model, loaders, epochs, lr, device, patience=3):
    """
    Trains the model with clean terminal blocks + tqdm bars, logs to TensorBoard,
    saves best checkpoint to models/resnet18_best.pt, and a curve PNG each epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    writer = SummaryWriter("runs/recycle")

    # ----- pretty header -----
    pstats = _num_params(model)
    classes_line = ", ".join(_class_counts(loaders["train"].dataset)) or "(classes unavailable)"
    print("\n" + "=" * 68)
    print(" TRAINING START ")
    print("-" * 68)
    print(f" epochs: {epochs} | lr: {lr:.2e} | patience: {patience} | bs: {loaders['train'].batch_size}")
    print(f" train batches: {len(loaders['train'])} | val batches: {len(loaders['val'])}")
    print(f" params: {pstats['trainable']:,} trainable / {pstats['total']:,} total")
    print(f" classes: {classes_line}")
    print("=" * 68 + "\n")

    best_val_acc, best_state, bad = 0.0, None, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()

        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, True,  ep, epochs)
        va_loss, va_acc = run_epoch(model, loaders["val"],   criterion, optimizer, device, False, ep, epochs)

        scheduler.step(va_acc)
        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)

        # TensorBoard scalars
        writer.add_scalar("loss/train", tr_loss, ep)
        writer.add_scalar("loss/val",   va_loss, ep)
        writer.add_scalar("acc/train",  tr_acc,  ep)
        writer.add_scalar("acc/val",    va_acc,  ep)
        writer.flush()

        improved = va_acc > best_val_acc
        if improved:
            best_val_acc, best_state, bad = va_acc, model.state_dict(), 0
            Path("models").mkdir(parents=True, exist_ok=True)
            torch.save(best_state, "models/resnet18_best.pt")
        else:
            bad += 1

        # neat epoch summary block
        print("\n" + "-" * 68)
        print(f" EPOCH {ep:02d}/{epochs:02d}  |  time { _fmt_time(time.time()-ep_t0) }  |  lr { _current_lrs(optimizer) }")
        print("-" * 68)
        print(f"   Train  •  loss {tr_loss:.4f}  |  acc {tr_acc:.4f}")
        print(f"   Val    •  loss {va_loss:.4f}  |  acc {va_acc:.4f}  |  "
              f"{'NEW BEST ✓' if improved else f'best {best_val_acc:.4f}'}")
        if improved:
            print("   Saved  •  models/resnet18_best.pt")
        if bad >= patience:
            print("-" * 68)
            print(f" EARLY STOP  •  no val acc improvement for {patience} epoch(s)")
            print("-" * 68)
            _save_curve(history, ep, best_val_acc)
            break

        _save_curve(history, ep, best_val_acc)

    # load best before returning
    if best_state is not None:
        model.load_state_dict(best_state)

    writer.close()
    print("\n" + "=" * 68)
    print(f" TRAINING DONE  •  best val acc = {best_val_acc:.4f}")
    print("=" * 68 + "\n")
    return model, best_val_acc


# ---------- visualisation of data transformations ----------
import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # Force Tkinter backend (interactive window)
import matplotlib.pyplot as plt
from model import build_model
from data import load_data, build_transforms
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # stop the terminal annoying message suppress TensorFlow warnings if any
from PIL import Image
#------------------- Show RAW AND TRANSFORMED Samples -------------------(BOTH TOGETHER IN ONE tab :D)
from torchvision import datasets, transforms

def show_raw_samples(data_root, class_names, num_samples=5):
    """Display raw images without any transforms."""
    raw_dataset = datasets.ImageFolder(
        os.path.join(data_root, "train"), 
        transform=transforms.ToTensor()  # just convert to tensor for plotting
    )

def show_transformed_samples(dataset, class_names, num_samples=5):
    """Display images after transformations (what model sees)."""


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
# ---------- visualisation ----------
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_ROOT = r"C:\Users\fearl\OneDrive\Desktop\recycle_material_classifier\data\images"
    MODEL_PATH = "models/resnet18_best.pt"
    BATCH_SIZE = 32

    print(f"Using device: {DEVICE}")
    print("Starting visulisation...")

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


    # Show raw + transformed samples (single window)
       # 1️⃣ Show raw images
    show_raw_samples(DATA_ROOT, class_names, num_samples=5)

    # 2️⃣ Show transformed images
    train_dataset = data["dsets"]["train"]
    show_transformed_samples(train_dataset, class_names, num_samples=5)

    train_dataset = data["dsets"]["train"]
    show_raw_and_transformed_samples(DATA_ROOT, train_dataset, class_names, num_samples=5)

    print("\nvisualisation of transformation complete.")