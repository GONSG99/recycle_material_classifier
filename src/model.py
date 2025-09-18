# src/model.py
import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes: int, freeze_backbone: bool, device: torch.device) -> nn.Module:
    """ResNet-18 transfer learning; replace final FC with num_classes."""
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1  # newer torchvision
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)         # older API

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    model.to(device)
    return model
