import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes: int, freeze_backbone: bool, device: torch.device) -> nn.Module:
    
    # Load a pre-trained ResNet18 model
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1  
        model = models.resnet18(weights=weights)

    # Older torchvision fallback
    except Exception:
        model = models.resnet18(pretrained=True)         

    # Freeze backbone layers if specified
    if freeze_backbone:

        for p in model.parameters(): # No gradients, no updates for any layer you froze
            p.requires_grad = False 

    # Replace the final fully connected layer
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    model.to(device)

    return model