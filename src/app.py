import json, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import gradio as gr
from model import build_model
import cv2
import threading
import time

stop_flag = False  # global flag to stop the thread

WEIGHTS = Path("models/resnet18_best.pt")
LABELS  = Path("models/labels.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(LABELS) as f:
    idx2name = {int(k): v for k, v in json.load(f).items()}
class_names = [idx2name[i] for i in sorted(idx2name.keys())]

model = build_model(num_classes=len(class_names), freeze_backbone=False, device=device)
state = torch.load(WEIGHTS, map_location=device)
model.load_state_dict(state)
model.eval()

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

<<<<<<< HEAD

# ---- IP Webcam setup ----
#ip_url = "http://10.132.39.118:8080/video"  # replace with your phone's IP
#cap = cv2.VideoCapture(ip_url)
#ret, prev = cap.read()
#prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

def live_ipcam_stream():
    global prev_gray
    ret, frame = cap.read()
    if not ret:
        return {"label":"No frame","conf":0}, {}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    motion_level = cv2.countNonZero(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1])
    prev_gray = gray

    if motion_level > 100:  # adjust threshold
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return predict(img)
    else:
        return {"label":"No motion","conf":0}, {}

stop_flag = False  # global flag to stop the thread

def start_live_feed():
    global stop_flag
    stop_flag = False
    def run():
        while not stop_flag:
            outputs = live_ipcam_stream()  # returns (json_dict, label_dict)
            json_out_live.update(outputs[0])
            label_out_live.update(outputs[1])
            time.sleep(0.1)
    threading.Thread(target=run, daemon=True).start()

def stop_live_feed():
    global stop_flag
    stop_flag = True

def live_ipcam_generator():
    global prev_gray
    while True:
        ret, frame = cap.read()
        if not ret:
            yield {"label": "No frame", "conf": 0}, {}
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        motion_level = cv2.countNonZero(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1])
        prev_gray = gray

        if motion_level > 100:  # motion threshold
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            yield predict(img)
        else:
            yield {"label": "No motion", "conf": 0}, {}

        time.sleep(0.1)  # pause 100ms between frames

=======
>>>>>>> e3250e4730da9544e4bd012789c9c4869bf00b54
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
            json_out = gr.JSON(label="Prediction (top class + confidence %)")
            label_out = gr.Label(num_top_classes=3, label="Top-3 probabilities")
            predict_btn.click(predict, inputs=img_input, outputs=[json_out, label_out])

        # --- Local Webcam ---
        # with gr.TabItem("Webcam"):
        #     webcam_input = gr.Camera(type="pil", label="Take a picture")
        #     json_out_cam = gr.JSON(label="Prediction (top class + confidence %)")
        #     label_out_cam = gr.Label(num_top_classes=3, label="Top-3 probabilities")
        #     webcam_input.change(predict, inputs=webcam_input, outputs=[json_out_cam, label_out_cam])

        # # --- Live IP Webcam ---
        # with gr.TabItem("Live IP Webcam"):
        #     json_out_live = gr.JSON(label="Prediction (top class + confidence %)")
        #     label_out_live = gr.Label(num_top_classes=3, label="Top-3 probabilities")
        #     start_btn = gr.Button("Start Live Feed")
        #     stop_btn  = gr.Button("Stop Live Feed")

        #     # Start the generator when the button is clicked
        #     start_btn.click(
        #         live_ipcam_generator,
        #         inputs=[],
        #         outputs=[json_out_live, label_out_live]
        #     )

    # Stop button can just close the browser tab or set a global stop flag

if __name__ == "__main__":
    demo.launch(inbrowser=True)
