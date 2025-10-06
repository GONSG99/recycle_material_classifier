import torch
import numpy as np
import cv2
from PIL import Image
from data import build_transforms
from model import build_model
import json
from pathlib import Path


def generate_gradcam(img, model, device, class_names, threshold=0.5):

    """
    Generate Grad-CAM for a given image and model.

    Args:
        img (PIL.Image): Input image.
        model (torch.nn.Module): Pretrained model.
        device (torch.device): Torch device (cpu/cuda).
        class_names (list): List of class labels.
        threshold (float): Threshold for masking heatmap.

    Returns:
        overlay_rgb (PIL.Image): Overlay of heatmap on original image.
        heatmap (np.ndarray): Heatmap array.
        pred_label (str): Predicted class label.
        confidence (float): Probability of predicted class.
    """

    # Load transforms
    _, eval_transforms = build_transforms(224)
    
    # Handle input type
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    elif not isinstance(img, Image.Image):
        raise ValueError("Input must be a file path or PIL.Image")
    
    img_tensor = eval_transforms(img).unsqueeze(0).to(device)

    # Hook storage
    feats, grads = [], []
    def fwd_hook(m, i, o): feats.append(o)
    def bwd_hook(m, gi, go): grads.append(go[0])

    # Register hooks
    layer = model.layer4[1].conv2
    layer.register_forward_hook(fwd_hook)
    layer.register_full_backward_hook(bwd_hook)

    # Forward pass
    out = model(img_tensor)
    pred_idx = out.argmax(1).item()
    score = out[0, pred_idx]

    # Confidence (softmax)
    probs = torch.softmax(out, dim=1)
    confidence = probs[0, pred_idx].item()

    # Backward pass
    model.zero_grad()
    score.backward()

    # Grad-CAM calculation
    grad = grads[0][0].detach().cpu().numpy()
    feat = feats[0][0].detach().cpu().numpy()
    weights = grad.mean(axis=(1, 2))
    cam = np.maximum(np.sum(weights[:, None, None] * feat, axis=0), 0)
    cam = cv2.resize(cam, img.size)
    cam = cam / cam.max()

    # Mask
    mask = (cam > threshold).astype(np.uint8) * 255

    # Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_TURBO)
    heatmap_masked = cv2.bitwise_and(heatmap, heatmap, mask=mask)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    orig_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig_bgr, 0.5, heatmap_masked, 0.5, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return Image.fromarray(overlay_rgb), heatmap_rgb, class_names[pred_idx], confidence


if __name__ == "__main__":
    
    # Load labels
    with open("models/labels.json", "r") as f:
        idx2name = {int(k): v for k, v in json.load(f).items()}
    class_names = [idx2name[i] for i in sorted(idx2name.keys())]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names), freeze_backbone=False, device=device)
    model.load_state_dict(torch.load("models/resnet18_best.pt", map_location=device))
    model.eval()
    print("Model loaded.")

    # Test with one image
    img = Image.open("data/own_images/test/paper/paper_normal_dark_23sec_013.jpg").convert("RGB")
    overlay, heatmap, pred_label, conf = generate_gradcam(img, model, device, class_names)

    print(f"Final Prediction: {pred_label} ({conf:.2%})")
    overlay.show()  # Display overlay
    heatmap_img = Image.fromarray(heatmap)
    heatmap_img.show()  # Display heatmap
    