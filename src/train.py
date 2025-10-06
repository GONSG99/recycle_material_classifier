# -------------------------------------------------------------
# This script handles the full training loop:
# 1️⃣ Train and validate a model for multiple epochs
# 2️⃣ Track accuracy, loss, learning rate, etc.
# 3️⃣ Save the best model (highest val accuracy)
# 4️⃣ Automatically stop early if val accuracy stops improving
# 5️⃣ Save training history and graphs
# -------------------------------------------------------------

import json, time
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from typing import Tuple, Dict, List
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# -------------------------------------------------------------
# HELPER: Compute batch accuracy
# -------------------------------------------------------------
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

# -------------------------------------------------------------
# HELPER: Format seconds to mm:ss (for clean epoch timing)
# -------------------------------------------------------------
def _fmt_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    return f"{m:02d}:{s:02d}"

# -------------------------------------------------------------
# HELPER: Get all target labels from a dataset
# -------------------------------------------------------------
def _get_targets(ds) -> torch.Tensor:
    if hasattr(ds, "targets"):
        return torch.as_tensor(ds.targets)
    return torch.as_tensor([t for _, t in ds.samples])

# -------------------------------------------------------------
# HELPER: Count how many samples per class
# -------------------------------------------------------------
def _class_counts(ds) -> List[str]:
    if not hasattr(ds, "classes"): return []
    y = _get_targets(ds)
    counts = torch.bincount(y, minlength=len(ds.classes)).tolist()
    return [f"{cls}({cnt})" for cls, cnt in zip(ds.classes, counts)]

# -------------------------------------------------------------
# HELPER: Count total and trainable parameters
# -------------------------------------------------------------
def _num_params(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

# -------------------------------------------------------------
# HELPER: Show current learning rate(s)
# -------------------------------------------------------------
def _current_lrs(optimizer) -> str:
    lrs = sorted({pg["lr"] for pg in optimizer.param_groups})
    if len(lrs) == 1: return f"{lrs[0]:.2e}"
    return ", ".join(f"{lr:.2e}" for lr in lrs)

# -------------------------------------------------------------
# FUNCTION: Run one full epoch (Train OR Validation)
# -------------------------------------------------------------
def run_epoch(model,loader,criterion,optimizer,device,train: bool,epoch: int,epochs: int,) -> Tuple[float, float]:
    
    model.train() if train else model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    total = 0

    # tqdm creates a nice progress bar in terminal
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

            out = model(xb)             # Forward pass 
            loss = criterion(out, yb)   # Compute loss

            if train:
                loss.backward()         # Backpropagation 
                optimizer.step()        # Update weights

        # Update running totals
        bsz = xb.size(0)
        epoch_loss += loss.item() * bsz
        epoch_acc  += accuracy_from_logits(out, yb) * bsz
        total      += bsz

        # Update progress bar text
        bar.set_postfix(loss=f"{epoch_loss/max(total,1):.4f}",
                        acc=f"{epoch_acc/max(total,1):.4f}")
        
    # Return average loss and accuracy
    return epoch_loss / total, epoch_acc / total


# -------------------------------------------------------------
# FUNCTION: Save training curves (loss + accuracy)
# -------------------------------------------------------------
def _save_final_figure(history, best_val_acc, outpath="reports/curves_final.png", title="ResNet (Fine-tune)"):

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    xs = range(1, len(history["train_loss"]) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- Loss subplot ----
    axs[0].plot(xs, history["train_loss"], label="train")
    axs[0].plot(xs, history["val_loss"],   label="val")
    axs[0].set_title("Loss"); axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss"); axs[0].legend()

     # ---- Accuracy subplot ----
    axs[1].plot(xs, history["train_acc"], label="train")
    axs[1].plot(xs, history["val_acc"],   label="val")
    axs[1].set_title("Accuracy"); axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Acc"); axs[1].legend()

    fig.suptitle(f"{title} | best val acc = {best_val_acc:.4f}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

    # Save training history as JSON
    with open("reports/history.json", "w") as f:
        json.dump(history, f, indent=2)


# -------------------------------------------------------------
# MAIN FUNCTION: train_model
# -------------------------------------------------------------
def train_model(model, loaders, epochs, lr, device, patience=3):

    # 1️⃣ Define loss, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    writer = SummaryWriter("runs/recycle") # For TensorBoard visualization

    # Print training setup summary
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

    # 2️⃣ Initialize trackers
    best_val_acc, best_state, bad = 0.0, None, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    saved_final = False

    # 3️⃣ Training loop over all epochs
    for ep in range(1, epochs + 1):
        ep_t0 = time.time()

        # Run training + validation
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, True,  ep, epochs)
        va_loss, va_acc = run_epoch(model, loaders["val"],   criterion, optimizer, device, False, ep, epochs)

        # Update scheduler based on validation accuracy
        scheduler.step(va_acc)

        # Store metrics in history
        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)

        # Log to TensorBoard
        writer.add_scalar("loss/train", tr_loss, ep)
        writer.add_scalar("loss/val",   va_loss, ep)
        writer.add_scalar("acc/train",  tr_acc,  ep)
        writer.add_scalar("acc/val",    va_acc,  ep)
        writer.flush()

        
        # Check for improvement
        improved = va_acc > best_val_acc
        if improved:
            best_val_acc, best_state, bad = va_acc, model.state_dict(), 0
            Path("models").mkdir(parents=True, exist_ok=True)
            torch.save(best_state, "models/resnet18_best.pt")
        else:
            bad += 1  # No improvement counter

        # Print summary for this epoch
        print("\n" + "-" * 68)
        print(f" EPOCH {ep:02d}/{epochs:02d}  |  time { _fmt_time(time.time()-ep_t0) }  |  lr { _current_lrs(optimizer) }")
        print("-" * 68)
        print(f"   Train  •  loss {tr_loss:.4f}  |  acc {tr_acc:.4f}")
        print(f"   Val    •  loss {va_loss:.4f}  |  acc {va_acc:.4f}  |  " f"{'NEW BEST ✓' if improved else f'best {best_val_acc:.4f}'}")
        
        if improved:
            print("   Saved  •  models/resnet18_best.pt")

        # Early stopping
        if bad >= patience:
            print("-" * 68)
            print(f" EARLY STOP  •  no val acc improvement for {patience} epoch(s)")
            print("-" * 68)
            _save_final_figure(history, best_val_acc, outpath="reports/curves_final.png", title="ResNet (Fine-tune)")
            saved_final = True
            break

    # 4️⃣ Load the best model weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # 5️⃣ Save final results if not done during early stop
    if not saved_final:
        _save_final_figure(history, best_val_acc, outpath="reports/curves_final.png", title="ResNet (Fine-tune)")

    writer.close()
    
    # 6️⃣ Summary printout
    print("\n" + "=" * 68)
    print(f" TRAINING DONE  •  best val acc = {best_val_acc:.4f}")
    print(" Saved final curves: reports/curves_final.png")
    print("=" * 68 + "\n")

    return model, best_val_acc
