# PPE-watch

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange?logo=gradio)
![License](https://img.shields.io/badge/License-MIT-green)
![Deployed](https://img.shields.io/badge/Status-Deployed%20in%20Factory-brightgreen)

Real-time Personal Protective Equipment (PPE) detection system built with YOLOv8 and Gradio. Deployed and running live in a factory environment.

Detects: **Helmets · Gloves · Eye Protection · Ear Protection · Masks** — and flags violations in real-time.

## Demo

<video src="https://github.com/user-attachments/assets/47bf975f-f39f-4389-839d-5f4520b35ad6" controls autoplay muted loop width="100%"></video>

## How It Works

Standard YOLO inference often produces noisy, conflicting detections — e.g. flagging both "Helmet" and "No Helmet" on the same head. This project adds a custom post-processing layer on top of raw model output to fix that:

**Per-class confidence thresholds** — each PPE class has a tuned minimum confidence based on how visually ambiguous it is. Masks require 75%+, gloves 92%+ (bare hands and gloves are nearly identical to a camera at distance).

**Mutual exclusion filtering** — pairs like `Helmet/No_Helmet`, `Gloves/No_Gloves` are treated as exclusive. When two conflicting detections overlap (measured by Intersection over Minimum Area, not standard IoU), the lower-confidence one is dropped. IoMin is used instead of IoU because a small Eye Protection box sitting inside a larger Person box would score near-zero on IoU but 1.0 on IoMin — the right call.

The result is stable, non-flickering detections suitable for a real production environment.

## Tech Stack

| Layer | Tool |
|---|---|
| Detection model | YOLOv8 (Ultralytics) |
| Web UI | Gradio |
| Image processing | OpenCV, NumPy |
| Edge deployment | Jetson Nano / Mac M3 |

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   python app.py
   ```

3. **Open in browser:**
   `http://127.0.0.1:7860`

## Deployment

This system is deployed and running in a real factory setting. The `best.pt` model weights are self-contained — no additional dataset files needed.

For Jetson Nano or Mac M3 deployment, see [JETSON_DEPLOY_GUIDE.md](JETSON_DEPLOY_GUIDE.md).
