"""
Master Exporter for Lossless Edge Optimization (FP32)
Exports to both ONNX and OpenVINO without quantizing the math.
"""

from pathlib import Path
from ultralytics import YOLO

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DETECTOR_PATH  = BASE_DIR / "models" / "corrosion_detector.pt"
GRADER_PATH    = BASE_DIR / "models" / "corrosion_grader.pt"

print(" Starting Lossless FP32 Conversions...")

# Load raw PyTorch models
detector = YOLO(str(DETECTOR_PATH))
grader = YOLO(str(GRADER_PATH))

# ---------------------------------------------------------
# 1. Export to OpenVINO (FP32 / Full Precision)
# ---------------------------------------------------------
print("\nCompiling OpenVINO FP32...")
# half=False keeps 32-bit math. imgsz=640 keeps the high-res grid.
detector.export(format="openvino", half=False, imgsz=640)
grader.export(format="openvino", half=False, imgsz=224)

# ---------------------------------------------------------
# 2. Export to ONNX (FP32 / Full Precision)
# ---------------------------------------------------------
print("\n Compiling ONNX FP32...")
detector.export(format="onnx", half=False, imgsz=640)
grader.export(format="onnx", half=False, imgsz=224)

print("\n All lossless models generated successfully in the models/ folder!")