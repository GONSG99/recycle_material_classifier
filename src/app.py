# src/app.py
import json, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import gradio as gr
from model import build_model
import cv2
import threading
import time
from explain import generate_gradcam

stop_flag = False  # global flag to stop the thread

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

# ---- prediction and heatmaps ----
def predict(img: Image.Image):
    # Grad-CAM
    overlay, heatmap, pred_label, conf = generate_gradcam(img, model, device, class_names)
    
    # Probabilities
    with torch.no_grad():
        x = tfm(img.convert("RGB")).unsqueeze(0).to(device)
        probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().tolist()
        scores = {cls: float(probs[i]) for i, cls in enumerate(class_names)}
        top = max(scores, key=scores.get)
    
    return [img, overlay, heatmap], pred_label, conf, scores

# ---- history ----
MAX_HISTORY = 12

def classify_and_update(img, history_state):
    if img is None:
        return [], "N/A", "N/A", {}, history_state
    
    # run classification
    gallery_imgs, pred_label, conf, all_scores = predict(img)
    
    # update history
    history_state.append(img)
    history_state = history_state[-MAX_HISTORY:]
    
    # pad with None for empty slots
    padded = history_state + [None]*(MAX_HISTORY - len(history_state))
    
    return gallery_imgs, pred_label, f"{round(conf*100)}%", all_scores, *padded, history_state

# ---- history select ----
def on_history_select(evt: gr.SelectData, history_state):
    return history_state[evt.index]

# ---- history click ----
def on_history_click(idx, history_state):
    if idx < len(history_state):
        return history_state[idx]
    return None

# ---- IP Webcam setup ----
# ip_url = "http://10.132.39.1:8080/video"  # replace with your phone's IP
#ip_url = "http://192.168.1.6:8080/video"
ip_url = "http://192.168.1.4:8080/video"

# ip_url = "http://10.132.39.1:8080/video"
# cap = None
# prev_gray = None

def start_live_feed():
    global stop_flag
    stop_flag = False
    def run():
        while not stop_flag:
            outputs = live_ipcam_generator()  # returns (json_dict, label_dict)
            json_out_live.update(outputs[0])
            label_out_live.update(outputs[1])
            time.sleep(0.1)
    threading.Thread(target=run, daemon=True).start()

def stop_live_feed():
    global stop_flag
    stop_flag = True

cap = None
prev_gray = None
motion_active = False
recent_preds = []

#ip_url = "http://10.132.39.1:8080/video"
#ip_url = "http://192.168.1.6:8080/video"
def live_ipcam_generator():
    """
    Generator that yields only frames with motion detected.
    Skips all frames without meaningful motion.
    """
    global cap, prev_gray

    motion_threshold = 100  # number of changed pixels to consider "motion"
    last_trigger_time = 0
    cooldown_sec = 0.5  # optional: avoid multiple triggers for same object

    while True:
        # Initialize camera if not already
        if cap is None or not cap.isOpened():
            try:
                cap = cv2.VideoCapture(ip_url)
                time.sleep(1)
                ret, prev = cap.read()
                if not ret or prev is None:
                    prev_gray = None
                    raise ValueError("No frame received")
                prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            except Exception:
                dummy_img = Image.new("RGB", (224, 224), (0, 0, 0))
                yield {"label": "Camera offline", "conf": 0}, {}, [dummy_img], {"motion_level": 0}
                time.sleep(1)
                continue

        # Read frame
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            cap = None
            dummy_img = Image.new("RGB", (224, 224), (0, 0, 0))
            yield {"label": "Camera disconnected", "conf": 0}, {}, [dummy_img], {"motion_level": 0}
            time.sleep(1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_level = cv2.countNonZero(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1])
        else:
            motion_level = 0

        prev_gray = gray

        # Only process frames with motion above threshold
        if motion_level > motion_threshold:
            current_time = time.time()
            if current_time - last_trigger_time >= cooldown_sec:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = img.resize((840, 480))
                pred_images, pred_label, conf, scores = predict(img)

                pred_json = {"label": pred_label, "conf": round(conf * 100, 2)}
                motion_info = {"motion_level": motion_level}

                last_trigger_time = current_time

                yield pred_json, scores, pred_images, motion_info
            else:
                # Skip frame due to cooldown
                continue
        else:
            # Skip frames without motion
            continue

        # tiny sleep to avoid hogging CPU
        time.sleep(0.01)

# ---- minimal CSS ----
css = """
footer, #footer, .footer, [data-testid="branding"] {display:none !important;}
a[href*="gradio.app"] {display:none !important;}
"""

# ---- New Gradio Blocks UI ----
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("<h1>♻️ Recycle Material Classifier</h1>")
    gr.Markdown("Upload a photo of a recyclable item to classify it as **paper**, **plastic**, or **metal**.")

    with gr.Tabs():
        # --- Upload Image ---
        # with gr.TabItem("Upload Image"):
        #     img_input = gr.Image(type="pil", label="  Upload an image")
        #     predict_btn = gr.Button("Predict")
        #     # Side-by-side gallery + bar chart
        #     gallery_out = gr.Gallery(label="Original & Grad-CAM", columns=2, height=300)
        #     label_out = gr.Label(num_top_classes=3, label="Top-3 probabilities")
            
        #     predict_btn.click(predict, inputs=img_input, outputs=[gallery_out, label_out])
        with gr.TabItem("Upload Image"):

            with gr.Row(variant="panel"):
                # --- Input Column ---
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil", 
                        label="Upload Image",
                        height=350
                    )
                    # Load initial history
                    history_state = gr.State([])
                    with gr.Row():
                        history_slots = [
                            gr.Image(type="pil", interactive=False, height=120, width=120, label=f"#{i+1}")
                            for i in range(MAX_HISTORY)
                        ]
                        
                        # Add select and click events to each history slot
                        for i, slot in enumerate(history_slots):
                            slot.select(
                                fn=lambda h, i=i: on_history_click(i, h), 
                                inputs=history_state, 
                                outputs=image_input
                            )

                # --- Output Column ---
                with gr.Column(scale=1):
                    gr.Markdown("<h2>Results</h2>")
                    predicted_label = gr.Textbox(label="Predicted Material", interactive=False)
                    confidence_score = gr.Textbox(label="Confidence", interactive=False)
                    all_scores_label = gr.Label(num_top_classes=3, label="All Confidence Scores")

                    # Add heatmap
                    heatmap_gallery = gr.Gallery(
                        label="Visualizations",
                        columns=3,
                        height=300
                    )
                    submit_btn = gr.Button("Classify", variant="primary")
                    
            # --- Button Logic ---
            submit_btn.click(
                fn=classify_and_update,
                inputs=[image_input, history_state],
                outputs=[heatmap_gallery, predicted_label, confidence_score, all_scores_label, *history_slots, history_state]
            )
            
         # --- Live IP Webcam ---
        with gr.TabItem("Live IP Webcam"):
            json_out_live = gr.JSON(label="Prediction (top class + confidence %)")
            label_out_live = gr.Label(num_top_classes=3, label="Top-3 probabilities")
            live_feed = gr.Gallery(label="Live Feed",
                                   height=500,     # adjust to fit your page
                                   columns=1       # 1 image per row
                                   )
            motion_out = gr.JSON(label="Motion Info")
            start_btn = gr.Button("Start Live Feed")
            stop_btn  = gr.Button("Stop Live Feed")

            start_btn.click(
                live_ipcam_generator,
                inputs=[],
                outputs=[json_out_live, label_out_live, live_feed, motion_out]
            )

    # Stop button can just close the browser tab or set a global stop flag

if __name__ == "__main__":
    demo.launch(inbrowser=True)
