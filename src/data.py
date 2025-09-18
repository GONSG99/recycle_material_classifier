# src/data.py
from pathlib import Path
import os, json, torch
from typing import Dict, Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf

def load_data(data_root: str, bs: int, num_workers: int = 2) -> Dict:
    """ImageFolder loader for data/images/train|val|test/{paper,plastic,metal}"""
    train_tf, eval_tf = build_transforms(224)
    dsets = {
        "train": datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf),
        "val":   datasets.ImageFolder(os.path.join(data_root, "val"),   transform=eval_tf),
        "test":  datasets.ImageFolder(os.path.join(data_root, "test"),  transform=eval_tf),
    }
    # Save idx->class for app/poster
    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/labels.json", "w") as f:
        json.dump({v: k for k, v in dsets["train"].class_to_idx.items()}, f, indent=2)

    pin = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(dsets["train"], batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=pin),
        "val":   DataLoader(dsets["val"],   batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin),
        "test":  DataLoader(dsets["test"],  batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin),
    }
    return {"dsets": dsets, "loaders": loaders}
