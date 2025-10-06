# -------------------------------------------------------------
# This script builds and returns your model architecture.
# You’re using ResNet-18 (a pre-trained CNN) as the backbone.
#
# Key idea:
#   - Load a pre-trained ResNet-18 (trained on ImageNet)
#   - Optionally freeze its layers (so they don’t update)
#   - Replace its final layer with your custom classifier
#   - Move the model to GPU or CPU
# -------------------------------------------------------------

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, freeze_backbone: bool, device: torch.device) -> nn.Module:
    
    # ---------------------------------------------------------
    # 1️⃣ Load a pre-trained ResNet-18
    # ---------------------------------------------------------
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1  # For newer torchvision versions
        model = models.resnet18(weights=weights)

    # Older torchvision fallback
    except Exception:
        model = models.resnet18(pretrained=True)         

    # ---------------------------------------------------------
    # 2️⃣ Optionally freeze the backbone (for transfer learning)
    # ---------------------------------------------------------
    # If freeze_backbone=True, only the last (new) layer will train
    if freeze_backbone:

        for p in model.parameters(): # No gradients, no updates for any layer you froze
            p.requires_grad = False 

    # ---------------------------------------------------------
    # 3️⃣ Replace the final fully-connected layer (the classifier)
    # ---------------------------------------------------------
    # The default ResNet18 outputs 1000 classes (ImageNet)
    # You’re replacing it with your own number of output classes

    in_feats = model.fc.in_features                 # Get input features of last layer      
    model.fc = nn.Linear(in_feats, num_classes)     # Replace with new layer

    # ---------------------------------------------------------
    # 4️⃣ Move model to selected device (GPU or CPU)
    # ---------------------------------------------------------
    model.to(device)
    
    return model    