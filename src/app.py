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

def predict(img: Image.Image):
    # Grad-CAM
    overlay, heatmap, pred_label, conf = generate_gradcam(img, model, device, class_names)
    
    # Probabilities
    with torch.no_grad():
        x = tfm(img.convert("RGB")).unsqueeze(0).to(device)
        probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().tolist()
        scores = {cls: float(probs[i]) for i, cls in enumerate(class_names)}
        top = max(scores, key=scores.get)
    
    return [img, overlay], scores


# # ---- IP Webcam setup ----
#ip_url = "http://10.132.39.1:8080/video"  # replace with your phone's IP
ip_url = "http://192.168.1.6:8080/video"
#ip_url = "http://10.132.39.1:8080/video"
cap = None
prev_gray = None


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
ip_url = "http://10.132.39.1:8080/video"
#ip_url = "http://192.168.1.6:8080/video"
def live_ipcam_generator():
    global cap, prev_gray

    while True:
        # Try to initialize camera if not already or if previous failed
        if cap is None or not cap.isOpened():
            try:
                cap = cv2.VideoCapture(ip_url)
                time.sleep(1)  # give the stream a second to start
                ret, prev = cap.read()
                if not ret or prev is None:
                    prev_gray = None
                    raise ValueError("No frame received")
                prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                # Yield placeholder outputs while camera is offline
                dummy_img = Image.new("RGB", (224,224), (0,0,0))
                yield {"label":"Camera offline", "conf":0}, {}, [dummy_img], {"motion":0}
                time.sleep(1)
                continue

        # Read frame
        ret, frame = cap.read()
        if not ret or frame is None:
            # Release the failed capture and try to reconnect next iteration
            cap.release()
            cap = None
            dummy_img = Image.new("RGB", (224,224), (0,0,0))
            yield {"label":"Camera disconnected", "conf":0}, {}, [dummy_img], {"motion":0}
            time.sleep(1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_level = cv2.countNonZero(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1])
        else:
            motion_level = 0
        prev_gray = gray

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Prediction only if motion detected
        if motion_level > 100:
            pred_images, scores = predict(img)
            pred_json = {"label": max(scores, key=scores.get), "conf": round(max(scores.values())*100,2)}
        else:
            pred_images, scores = [img], {}
            pred_json = {"label":"No motion", "conf":0}

        motion_info = {"motion_level": motion_level}

        # Yield outputs in exact order for Gradio
        yield pred_json, scores, pred_images, motion_info

        time.sleep(0.03)


# ---- minimal CSS: hide branding/footer ----
css = """
footer, #footer, .footer, [data-testid="branding"] {display:none !important;}
a[href*="gradio.app"] {display:none !important;}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Recycle Material Classifier")
    gr.Markdown("Upload an image or use live camera below:")

    with gr.Tabs():
        # --- Upload Image ---
        with gr.TabItem("Upload Image"):
            img_input = gr.Image(type="pil", label="Upload an image")
            predict_btn = gr.Button("Predict")
            # Side-by-side gallery + bar chart
            gallery_out = gr.Gallery(label="Original & Grad-CAM", columns=2, height=300)
            label_out = gr.Label(num_top_classes=3, label="Top-3 probabilities")
            
            predict_btn.click(predict, inputs=img_input, outputs=[gallery_out, label_out])


        # --- Live IP Webcam ---
        with gr.TabItem("Live IP Webcam"):
            json_out_live = gr.JSON(label="Prediction (top class + confidence %)")
            label_out_live = gr.Label(num_top_classes=3, label="Top-3 probabilities")
            live_feed = gr.Gallery(label="Live Feed")
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
