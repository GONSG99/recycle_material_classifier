# src/app.py
import json, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import gradio as gr
from model import build_model

# ---- paths ----
WEIGHTS = Path("models/resnet18_best.pt")
LABELS  = Path("models/labels.json")

# ---- device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- labels ----
with open(LABELS) as f:
    idx2name = {int(k): v for k, v in json.load(f).items()}
class_names = [idx2name[i] for i in sorted(idx2name.keys())]

# ---- model ----
model = build_model(num_classes=len(class_names), freeze_backbone=False, device=device)
state = torch.load(WEIGHTS, map_location=device)
model.load_state_dict(state)
model.eval()

# ---- transforms (same as eval) ----
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

@torch.inference_mode()
def predict(img: Image.Image):
    x = tfm(img.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().tolist()
    scores = {cls: float(probs[i]) for i, cls in enumerate(class_names)}
    top = max(scores, key=scores.get)
    return {"label": top, "conf": round(scores[top]*100, 2)}, scores

# ---- minimal CSS: hide branding/footer ----
css = """
footer, #footer, .footer, [data-testid="branding"] {display:none !important;}
a[href*="gradio.app"] {display:none !important;}
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.JSON(label="Prediction (top class + confidence %)"),
        gr.Label(num_top_classes=3, label="Top-3 probabilities")
    ],
    title="Recycle Material Classifier",
    description="Upload a photo → get paper / plastic / metal + confidence",
    allow_flagging="never",
    css=css,
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
