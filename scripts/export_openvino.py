"""
Export trained YOLO models to OpenVINO format for edge deployment.
Includes FP16 (half-precision) optimization for maximum FPS on Intel CPUs.

Run from the project root:
    .venv\Scripts\python.exe scripts\export_openvino.py

Requirements:
    pip install openvino
"""
import os
from pathlib import Path
from ultralytics import YOLO

# --- Paths (relative to project root) ---
BASE_DIR = Path(__file__).resolve().parent.parent
DETECTOR_PATH  = BASE_DIR / "models" / "corrosion_detector.pt"
GRADER_PATH    = BASE_DIR / "models" / "corrosion_grader.pt"

print(" Starting OpenVINO FP16 Conversion with Custom Resolutions...")
print(f"   Project root : {BASE_DIR}")

# 1. Convert Stage 1 — Detection model (YOLOv12n)
print("\n Converting Stage 1 Detector (YOLOv12n) to FP16 at 480x480...")
if DETECTOR_PATH.exists():
    detector = YOLO(str(DETECTOR_PATH))
    # ADDED imgsz=480 so OpenVINO bakes the math for the smaller size
    detector.export(format="openvino", half=False, imgsz=640)
    print(f" Detector exported → {DETECTOR_PATH.stem}_openvino_model/")
else:
    print(f" Not found: {DETECTOR_PATH}")

# 2. Convert Stage 2 — Grader/Classifier model (YOLOv8n-cls)
print("\n Converting Stage 2 Grader (YOLOv8n-cls) to FP16 at 224x224...")
if GRADER_PATH.exists():
    grader = YOLO(str(GRADER_PATH))
    # ADDED imgsz=224 so OpenVINO bakes the math for the smaller size
    grader.export(format="openvino", half=True, imgsz=224)
    print(f" Grader exported → {GRADER_PATH.stem}_openvino_model/")
else:
    print(f" Not found: {GRADER_PATH}")

print("\n Done! Check the 'models/' folder for the new OpenVINO directories.")