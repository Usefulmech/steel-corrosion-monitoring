# Developer Implementation Guide

This document is the practical, step-by-step guide for developers running, training, and deploying the cascaded YOLOv12 + YOLOv8-cls corrosion monitoring system.

## 1. Local Environment Setup

Ensure you have Python 3.8+ installed on your deployment machine (e.g., Lenovo T470s).

```bash
# Create and activate virtual environment
python -m venv corrosion_env
# Windows:
corrosion_env\Scripts\activate
# Linux/Mac:
source corrosion_env/bin/activate

# Install core dependencies
pip install ultralytics opencv-python numpy matplotlib requests
```

### Project Structure
```text
corrosion-monitoring/
├── models/
│   ├── detection_best.pt       # Stage 1: YOLOv12n
│   └── classification_best.pt  # Stage 2: YOLOv8n-cls
├── src/
│   ├── cascade_inference.py    # Core inference engine
│   ├── live_demo.py            # Webcam demo
│   └── rtsp_demo.py            # CCTV deployment script
├── frontend/                   # React/Vite dashboard
└── app.py                      # Flask backend API
```

## 2. Dataset Engineering & Roboflow Workflow

1. **Collect Custom Data**: Use a mobile phone (recommended for varied angles/lighting) and extract frames from CCTV footage using `extract_frames.py`. Do NOT annotate raw videos.
2. **Roboflow Projects**: Create two separate projects in Roboflow:
   * `corrosion-detection` (Object Detection - 1 class: "corrosion")
   * `corrosion-grading` (Classification - 4 classes: Grade-0 to Grade-3)
3. **Merge Public Datasets**: Import public datasets via Roboflow Universe to supplement your custom data. Map all public classes (e.g., "rust", "oxidation") to the single "corrosion" class.
4. **Preprocessing & Augmentation**:
   * Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
   * Detection: Resize 640x640, Rotation ±15°, Brightness ±15%, Mosaic.
   * Classification: Resize 224x224, Rotation ±20°, Brightness ±20%.

## 3. Training Pipeline (Google Colab / Kaggle)

The system utilizes YOLOv12n for object detection and YOLOv8n-cls for severity classification.

```python
# Install requirements
!pip install ultralytics roboflow -q

# 1. Download YOLOv12n Weights Manually
import urllib.request
urllib.request.urlretrieve("https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12n.pt", "yolov12n.pt")

# 2. Download Datasets from Roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
dataset_detect = rf.workspace("WORKSPACE").project("corrosion-detection").version(1).download("yolov8")
dataset_classify = rf.workspace("WORKSPACE").project("corrosion-grading").version(1).download("folder")

# 3. Train Detection (Stage 1)
from ultralytics import YOLO
model_detect = YOLO("yolov12n.pt")
model_detect.train(data=f"{dataset_detect.location}/data.yaml", epochs=100, imgsz=640, project='runs/detect')

# 4. Train Classification (Stage 2)
model_classify = YOLO("yolov8n-cls.pt") # Auto-downloads pretrained weights
model_classify.train(data=dataset_classify.location, epochs=100, imgsz=224, project='runs/classify')
```

## 4. CCTV Deployment (RTSP)

For deployment on your ASECAM 8MP 3.6MM 4K H.265 ONVIF IP Camera, use the RTSP protocol. Ensure the camera is powered via the 48V DC / 12V 2A PoE Adapter using an Ethernet cable connected to your local network.

### Common RTSP URL Structure
* ASECAM Main Stream: `rtsp://admin:password@<IP_ADDRESS>:554/stream1` (Check your specific IP configuration)
* ASECAM Sub Stream (Faster for low latency): `rtsp://admin:password@<IP_ADDRESS>:554/stream2`

### Minimal RTSP Monitor Script (`rtsp_demo.py`)
```python
import cv2
from ultralytics import YOLO

detector = YOLO("models/detection_best.pt")
classifier = YOLO("models/classification_best.pt")

# Note: Using sub-stream /102 for lower latency
cap = cv2.VideoCapture("rtsp://admin:password@192.168.1.64:554/Streaming/Channels/102")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce lag

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Stage 1: Detect
    detections = detector.predict(frame, conf=0.5, verbose=False)[0]
    
    # Stage 2: Classify
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = frame[y1:y2, x1:x2]
        
        results = classifier.predict(roi, verbose=False)[0]
        grade = results.probs.top1
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Grade-{grade}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
```

## 5. Troubleshooting & Optimization

* **High Latency/Lag:** Switch to the camera's sub-stream (`/Channels/102`). Ensure `cv2.CAP_PROP_BUFFERSIZE` is set to 1. Export the PyTorch models to ONNX (`model.export(format='onnx')`) for a ~30% CPU speedup.
* **Connection Timeout:** Verify IP address using the HikVision SADP tool. Ping the camera from the T470s command line. Ensure port 554 is not blocked by the firewall.
* **Low Classification Accuracy:** Collect more diverse mobile phone photos. Ensure classes are strictly balanced before augmenting. Increase training epochs to 150.
