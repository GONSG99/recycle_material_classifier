# src/app.py
import json, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import gradio as gr
from model import build_model

# ---- paths ----
EXAMPLES = [
    ["examples/paper.jpg"],
    ["examples/plastic.jpg"],
    ["examples/metal.jpg"],
]
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

def classify_image(img: Image.Image):
    """Wrapper function to format predictions for the new Gradio UI."""
    if img is None:
        return "N/A", "N/A", {}
        
    top_pred, all_scores = predict(img)
    
    # Return values formatted for the new output components
    return top_pred["label"], f"{top_pred['conf']}%", all_scores

# ---- minimal CSS: hide branding/footer ----
css = """
footer, #footer, .footer, [data-testid="branding"] {display:none !important;}
a[href*="gradio.app"] {display:none !important;}
"""

# ---- New Gradio Blocks UI ----
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("<h1>♻️ Recycle Material Classifier</h1>")
    gr.Markdown("Upload a photo of a recyclable item to classify it as **paper**, **plastic**, or **metal**.")
    
    with gr.Row(variant="panel"):
        # --- Input Column ---
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil", 
                label="Upload Image",
                height=350
            )
            gr.Examples(
                examples=EXAMPLES,
                inputs=image_input,
                label="Click an example to try it out!"
            )
            submit_btn = gr.Button("Classify", variant="primary")

        # --- Output Column ---
        with gr.Column(scale=1):
            gr.Markdown("<h2>Results</h2>")
            predicted_label = gr.Textbox(label="Predicted Material", interactive=False)
            confidence_score = gr.Textbox(label="Confidence", interactive=False)
            all_scores_label = gr.Label(num_top_classes=3, label="All Confidence Scores")

    # --- Button Logic ---
    submit_btn.click(
        fn=classify_image,
        inputs=image_input,
        outputs=[predicted_label, confidence_score, all_scores_label]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
