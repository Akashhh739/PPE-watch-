import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

import os

# Load the trained YOLO model (ensure it looks in the same folder as this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'best.pt')
model = YOLO(model_path)

def predict_image(img):
    """
    Run YOLO inference on the uploaded image or webcam frame
    """
    if img is None:
        return None
    
    try:
        # Gradio provides the image in RGB format, but YOLO expects BGR internally 
        # for its plotting colors to work correctly.
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Run inference with improved parameters
        results = model(img_bgr, conf=0.50, iou=0.45, max_det=20)
        
        # --- Custom Post-Processing Filter ---
        # 1. We want to remove mutually exclusive overlapping boxes
        #    e.g. You can't have 'Eye_Protection' and 'No_Eye_Protection' in the same spot.
        # 2. We want to apply higher thresholds to noisy classes like Masks and Gloves.
        
        boxes = results[0].boxes
        
        # Convert to a format we can easily manipulate
        # Format: [cls_id, conf, x1, y1, x2, y2]
        detections = []
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                
                # Filter 1: Apply stricter confidence to noisy classes
                cls_name = model.names[cls_id]
                
                # Drop specific classes entirely based on user request
                if cls_name == 'No_Eye_Protection': continue
                
                # Masks are notoriously difficult without high res, bump their reqs up
                if cls_name == 'Mask' and conf < 0.75: continue
                
                # --- The Glove Bias Hack ---
                # Bare hands ("No_Gloves") and tight gloves often look identical to AI.
                # To prevent confusing flickers, we apply extreme bias towards "No_Gloves".
                # If YOLO sees a hand, we assume it's bare (No_Gloves) unless it is 92% SURE it's a glove.
                if cls_name == 'Gloves' and conf < 0.92: continue 
                if cls_name == 'No_Gloves' and conf < 0.50: continue
                # ---------------------------
                
                # Eye/Ear protection often flicker or overlap with each other, demand higher confidence
                if 'Protection' in cls_name and conf < 0.65: continue
                
                # Helmets often get confused with human hair/heads
                if 'Helmet' in cls_name and conf < 0.65: continue
                
                detections.append({'id': i, 'cls_id': cls_id, 'cls_name': cls_name, 'conf': conf, 'box': [x1, y1, x2, y2]})
        
        # Filter 2: Mutually exclusive overlapping classes
        # Instead of strict IoU (Intersection over Union), we use IoMin (Intersection over Minimum Area).
        # This is because a small Eye_Protection box might be perfectly inside a large Person box.
        # We also want to suppress conflicting classes that overlap significantly.
        
        exclusive_sets = [
            {'Helmet', 'No_Helmet'},
            {'Eye_Protection', 'No_Eye_Protection'},
            {'Ear_Protection', 'No_Ear_Protection'},
            {'Gloves', 'No_Gloves'}
        ]
        
        def compute_iomin(boxA, boxB):
            """Compute Intersection over Minimum Area. 
            If a small box is 100% inside a big box, this returns 1.0"""
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            
            if interArea == 0: return 0.0
            
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            minArea = min(boxAArea, boxBArea)
            
            if minArea == 0: return 0.0
            return interArea / float(minArea)

        to_remove = set()
        for i in range(len(detections)):
            if i in to_remove: continue
            for j in range(i + 1, len(detections)):
                if j in to_remove: continue
                
                det1 = detections[i]
                det2 = detections[j]
                
                # Check if they belong to any mutually exclusive sets
                are_exclusive = False
                for ex_set in exclusive_sets:
                    if det1['cls_name'] in ex_set and det2['cls_name'] in ex_set:
                        are_exclusive = True
                        break
                        
                if are_exclusive:
                    # Check if they physically overlap (Intersection over Minimum area)
                    overlap = compute_iomin(det1['box'], det2['box'])
                    
                    # If they overlap by even 10%, one of them has to go
                    if overlap > 0.10: 
                        if det1['conf'] > det2['conf']:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            
        # Filter the original boxes object by keeping only the valid indices
        valid_indices = [det['id'] for i, det in enumerate(detections) if i not in to_remove]
        
        if len(valid_indices) > 0:
            results[0].boxes = results[0].boxes[valid_indices]
        else:
            # Empty out the boxes if everything was filtered
            results[0].boxes = results[0].boxes[[]] # empty tensor trick

        # --- End Custom Post-Processing ---
        
        print("\n" + "="*30)
        print("INFERENCE PREDICTION RESULTS:")
        found_objects = False
        for box in results[0].boxes:
            found_objects = True
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            print(f" -> Detected: {cls_name} (Confidence: {conf:.2f})")
            
        if not found_objects:
            print(" -> No PPE or objects detected.")
        print("="*30 + "\n")
        
        # Extract the image with the plotted bounding boxes natively from ultralytics
        # Make the lines slightly thicker and text more legible. 
        res_plotted = results[0].plot(line_width=2, font_size=1, conf=True)
        
        # When Gradio gives us the webcam frame, it's RGB.
        # YOLO .plot() returns a BGR formatted image. We convert it back to RGB for Gradio.
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        return res_rgb
    except Exception as e:
        import traceback
        print(f"\nERROR DURING INFERENCE:\n{traceback.format_exc()}\n")
        # Return original image to avoid breaking the UI completely
        return img

# Create the Gradio interface
with gr.Blocks(title="YOLOv8 Object Detection") as demo:
    gr.Markdown("# YOLOv8 Object Detection")
    gr.Markdown("Upload an image or use your webcam to run inference using your newly trained model.")
    
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Webcam Live Stream"):
                    # Use streaming=True for live webcam feed. This sends frames continuously
                    # to the server and avoids the snapshot button bug.
                    webcam_image = gr.Image(label="Live Webcam", sources=["webcam"], streaming=True, type="numpy")
                    gr.Markdown("*Live stream inference starts automatically based on camera input.*")
                with gr.Tab("Upload Image"):
                    upload_image = gr.Image(label="Upload Image", sources=["upload"], type="numpy")
                    predict_upload_btn = gr.Button("Run Image Inference", variant="primary")
        with gr.Column():
            # Output image component for the detected boxes
            output_image = gr.Image(label="Detection Output")
            
    # Link the inputs to the prediction function
    # For webcam stream, it automatically triggers inference on every incoming frame
    webcam_image.stream(fn=predict_image, inputs=webcam_image, outputs=output_image)
    
    # For upload, we still use the button click
    predict_upload_btn.click(fn=predict_image, inputs=upload_image, outputs=output_image)

if __name__ == "__main__":
    # Launch the Gradio app on the local network. 
    # Not specifying a hardcoded server_port allows Gradio to automatically pick 
    # the next free port (7861, 7862, etc.) if 7860 is already in use.
    demo.launch(server_name="127.0.0.1", share=False)
