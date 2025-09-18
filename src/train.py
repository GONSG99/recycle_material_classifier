# src/train.py
import torch
import torch.nn as nn
from typing import Tuple

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> Tuple[float, float]:
    model.train() if train else model.eval()
    epoch_loss = 0.0; epoch_acc = 0.0; total = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if train: optimizer.zero_grad()
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

    return epoch_loss / total, epoch_acc / total

def train_model(model, loaders, epochs, lr, device, patience=3):
    """Lab-2 style loop + ReduceLROnPlateau + early stop + best-weights save."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_val_acc, best_state, bad = 0.0, None, 0

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, loaders["val"],   criterion, optimizer, device, train=False)
        scheduler.step(va_acc)

        print(f"[{ep:02d}/{epochs}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc, best_state, bad = va_acc, model.state_dict(), 0
            torch.save(best_state, "models/resnet18_best.pt")
            print("  ↳ saved: models/resnet18_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print("  ↳ early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc
