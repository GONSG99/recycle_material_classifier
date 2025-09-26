        webcam_input = gr.Camera(label="Take a picture", type="pil")
            json_output2 = gr.JSON(label="Prediction (top class + confidence %)") 
            label_output2 = gr.Label(num_top_classes=3, label="Top-3 probabilities")
            webcam_input.change(predict, inputs=webcam_input, outputs=[json_output2, label_output2])