# QUICK REFERENCE CHEAT SHEET
## Steel Corrosion Monitoring System

---

## DATA COLLECTION - MOBILE vs CCTV

### ✅ RECOMMENDED: Mobile Phone for Training Data
```
Why mobile phone is PERFECT:
- Better quality than most CCTV cameras
- Flexible angles and distances
- Easy close-up detail capture
- Models generalize well

Collection Tips:
- Take 500-1000 photos
- Vary: angles, distances, lighting
- Multiple shots per corroded area (3-5)
- Include some blurry/low-quality shots
```

### Frame Extraction (If Using Video)
```python
# Extract frames from CCTV video - DO NOT annotate videos!
import cv2, os

def extract_frames(video_path, output_dir, frame_interval=60):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count, saved = 0, 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_count % frame_interval == 0:
            cv2.imwrite(f"{output_dir}/frame_{saved:04d}.jpg", frame)
            saved += 1
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved} frames")

extract_frames("cctv.mp4", "frames", 60)  # 1 frame/2sec at 30fps
```

---

## ROBOFLOW - KEY STEPS

### Model Architecture (Updated!)
```
Detection (Stage 1): YOLOv12n
  ✅ Latest architecture with Area Attention
  ✅ R-ELAN blocks for stability
  
Classification (Stage 2): YOLOv8n-cls (CHANGED!)
  ✅ Proven, stable, well-documented
  ✅ Auto-downloads pretrained weights
  ✅ Easier to train than YOLOv12-cls
  ✅ Excellent classification performance
```

### Dataset Strategy
```
Custom Data: 500-1000 images (30%)
  - Mobile phone photos
  - CCTV extracted frames
  
Public Data: 1500-2500 images (70%)
  - Roboflow Universe datasets
  - Search: "corrosion", "rust", "steel corrosion"
  
Total: 2000-3500 images
```

### Merging Public Datasets (3 Methods)

**Method 1: Direct Import in Roboflow (EASIEST)**
```
1. Create your project: "Corrosion-Detection"
2. Upload YOUR custom images first
3. Annotate your images
4. Find public dataset on Roboflow Universe
5. Copy dataset URL
6. Your project → "Add Images" → "Import from Roboflow"
7. Paste URL
8. Map classes: "rust" → "corrosion", etc.
9. Review imported images
10. Delete poor quality, fix annotations
```

**Method 2: Download + Manual Merge**
```python
from roboflow import Roboflow

# Download public dataset
rf = Roboflow(api_key="YOUR_KEY")
public = rf.workspace("public-ws").project("corrosion-data")
dataset = public.version(1).download("yolov8")

# Merge using merge_datasets.py script (see full guide)
# Then re-upload merged dataset to your Roboflow project
```

**Method 3: Class Mapping Example**
```
When importing public datasets, map all to "corrosion":

Public Dataset Classes → Your Class
"rust"               → "corrosion"
"oxidation"          → "corrosion"  
"light-corrosion"    → "corrosion"
"heavy-corrosion"    → "corrosion"
"corrosion"          → "corrosion"

All become single "corrosion" class for detection
```

### Detection Project Setup
```
1. Create project → Object Detection
2. Class: "corrosion"
3. **Upload custom images FIRST, then import public datasets**
4. Preprocessing: Resize 640x640, CLAHE
5. Augmentation: Rotation ±15°, Brightness ±15%, Mosaic, 3x multiplier
6. Split: 70/20/10
7. **Quality Check after merging:**
   - Review 50-100 random images
   - Remove duplicates
   - Fix incorrect annotations
   - Delete poor quality images
8. Export: YOLOv8 format
```

### Classification Project Setup
```
1. Create project → Classification
2. Classes: Grade-0, Grade-1, Grade-2, Grade-3
3. Preprocessing: Resize 224x224, CLAHE
4. Augmentation: Rotation ±20°, Brightness ±20%, 2x multiplier
5. Split: 70/20/10
6. Export: Folder Structure
```

---

## GOOGLE COLAB - ESSENTIAL CODE

### Complete Training Notebook

```python
# SETUP (NO WGET NEEDED!)
!pip install ultralytics roboflow -q
import os
import urllib.request

# DOWNLOAD WEIGHTS
# Detection: YOLOv12n (manual download)
detection_url = "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12n.pt"
if not os.path.exists("yolov12n.pt"):
    print("📥 Downloading YOLOv12n...")
    urllib.request.urlretrieve(detection_url, "yolov12n.pt")
    print("✅ Downloaded!")

# Classification: YOLOv8n-cls (auto-downloads on first use)
print("📋 YOLOv8n-cls will auto-download when training starts")

# LOAD DATASETS
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")

# Detection dataset
project_detect = rf.workspace("WORKSPACE").project("corrosion-detection")
dataset_detect = project_detect.version(1).download("yolov8")

# Classification dataset
project_classify = rf.workspace("WORKSPACE").project("corrosion-grading")
dataset_classify = project_classify.version(1).download("folder")

# TRAIN DETECTION (YOLOv12n)
from ultralytics import YOLO
model_detect = YOLO("yolov12n.pt")

model_detect.train(
    data=f"{dataset_detect.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    patience=50,
    project='runs/detect',
    name='corrosion_detection'
)

# TRAIN CLASSIFICATION (YOLOv8n-cls) ← CHANGED FROM YOLOv12
model_classify = YOLO("yolov8n-cls.pt")  # Auto-downloads pretrained weights

model_classify.train(
    data=dataset_classify.location,
    epochs=100,
    imgsz=224,
    batch=32,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    dropout=0.2,
    patience=50,
    project='runs/classify',
    name='corrosion_grading'
)

# DOWNLOAD MODELS
import shutil
os.makedirs("trained_models", exist_ok=True)
shutil.copy("runs/detect/corrosion_detection/weights/best.pt", "trained_models/detection_best.pt")
shutil.copy("runs/classify/corrosion_grading/weights/best.pt", "trained_models/classification_best.pt")
shutil.make_archive("trained_models", 'zip', "trained_models")

print("✅ Training complete!")
print("   Detection: YOLOv12n")
print("   Classification: YOLOv8n-cls")
```

---

## T470S DEPLOYMENT - QUICK START

### Setup Environment
```bash
python -m venv corrosion_env
corrosion_env\Scripts\activate  # Windows
pip install ultralytics opencv-python numpy matplotlib
```

### Project Structure
```
corrosion-monitoring/
├── models/
│   ├── detection_best.pt
│   └── classification_best.pt
├── test_images/
├── output/
├── cascade_inference.py
├── live_demo.py
└── rtsp_demo.py  ← For CCTV deployment
```

### Run Single Image
```bash
python cascade_inference.py
```

### Run Live Demo
```bash
python live_demo.py
```

### Run RTSP CCTV Monitor
```bash
python rtsp_demo.py
```

---

## RTSP CAMERA SETUP (8MP CCTV)

### Common RTSP URLs
```
HikVision (your 8MP camera):
rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101

Main stream (high quality): /Channels/101
Sub stream (faster): /Channels/102

Dahua:
rtsp://admin:password@IP:554/cam/realmonitor?channel=1&subtype=0

Generic:
rtsp://username:password@ip_address:port/stream_path
```

### Test RTSP Connection
```python
import cv2

rtsp_url = "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101"
cap = cv2.VideoCapture(rtsp_url)

if cap.isOpened():
    ret, frame = cap.read()
    cv2.imwrite("test.jpg", frame)
    print("✅ Connected!")
else:
    print("❌ Failed - check URL/credentials")
cap.release()
```

### Quick RTSP Monitor
```python
from ultralytics import YOLO
import cv2

detector = YOLO("models/detection_best.pt")
classifier = YOLO("models/classification_best.pt")

rtsp_url = "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101"
cap = cv2.VideoCapture(rtsp_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Detect
    detections = detector.predict(frame, conf=0.5)[0]
    
    # Classify each
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = frame[y1:y2, x1:x2]
        
        results = classifier.predict(roi)[0]
        grade = results.probs.top1
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Grade-{grade}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
```

---

## MINIMAL CASCADE CODE (Copy-Paste Ready)

```python
from ultralytics import YOLO
import cv2

# Load models
# Detection: YOLOv12n, Classification: YOLOv8n-cls
detector = YOLO("models/detection_best.pt")
classifier = YOLO("models/classification_best.pt")

# Load image
image = cv2.imread("test.jpg")

# Stage 1: Detect (YOLOv12n)
detections = detector.predict(image, conf=0.5)[0]

# Stage 2: Classify each detection (YOLOv8n-cls)
for box in detections.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    roi = image[y1:y2, x1:x2]
    
    results = classifier.predict(roi)[0]
    grade = results.probs.top1
    confidence = results.probs.top1conf.item()
    
    print(f"Grade-{grade}, Confidence: {confidence:.2f}")
    
    # Draw
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"Grade-{grade}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite("output.jpg", image)
```

---

## GRADING CRITERIA (ASTM D610 Based)

| Grade | Description | Rust Coverage |
|-------|-------------|---------------|
| **0** | Healthy | 0% |
| **1** | Mild | 0.01% - 0.03% |
| **2** | Moderate | 0.03% - 1.0% |
| **3** | Severe | > 1.0% |

---

## TROUBLESHOOTING

### Model not loading
```python
# Check file exists
import os
print(os.path.exists("models/detection_best.pt"))

# Verify PyTorch/CUDA
import torch
print(torch.cuda.is_available())
```

### Slow inference
```python
# Use smaller image size
results = model.predict(image, imgsz=416)

# Export to ONNX
model.export(format='onnx')
model_onnx = YOLO("model.onnx")
```

### RTSP Connection Issues
```
Problem: Connection timeout
→ Ping camera IP, check network

Problem: Authentication failed  
→ Verify username/password in camera settings

Problem: Port blocked
→ Check firewall, ensure port 554 is open

Problem: Stream not found
→ Try /Channels/102 (sub-stream) instead of /Channels/101

Problem: Lag/high latency
→ Use sub-stream or reduce buffer: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

Problem: Can't find camera IP
→ Use HikVision SADP tool or check router
```

### Low accuracy
- Collect more training data
- Increase epochs to 150-200
- Review annotations for consistency
- Add more augmentation

---

## EVALUATION METRICS

```python
# Detection model
metrics_detect = model_detect.val()
print(f"mAP50: {metrics_detect.box.map50:.4f}")
print(f"Precision: {metrics_detect.box.p:.4f}")
print(f"Recall: {metrics_detect.box.r:.4f}")

# Classification model
metrics_classify = model_classify.val()
print(f"Accuracy: {metrics_classify.top1:.4f}")
```

---

## SUCCESS TARGETS

✅ Detection mAP50 > 0.75
✅ Classification Accuracy > 0.80
✅ Inference < 200ms per image
✅ Live demo 5-10 FPS
✅ RTSP stream stable connection

---

## IMPORTANT NOTES

### Training vs Deployment
```
TRAINING DATA COLLECTION:
✅ Mobile phone photos (RECOMMENDED)
✅ Extract frames from CCTV video
✅ Mix both sources

DEPLOYMENT:
✅ 8MP CCTV via RTSP
✅ Real-time monitoring on T470s
```

### Key Principles
1. **NEVER annotate videos** - extract frames first, then annotate images
2. **Mobile phone is FINE** for training - models generalize well
3. **ALWAYS apply CLAHE** preprocessing for tropical conditions
4. **CONSISTENT annotation** is critical for accuracy
5. **BALANCE dataset** across all grades (0-3)
6. **VALIDATE regularly** during training
7. **TEST RTSP connection** before full deployment
8. **USE YOLOv8n-cls for classification** - more stable than YOLOv12-cls

### Model Combination Benefits
```
YOLOv12n (Detection) + YOLOv8n-cls (Classification) = BEST RESULTS

Why this combo?
✅ YOLOv12n: Cutting-edge detection with Area Attention
✅ YOLOv8n-cls: Battle-tested, proven classification
✅ Easy to train and deploy
✅ No accuracy loss - potentially better performance!
✅ YOLOv12-cls weights hard to get, YOLOv8-cls auto-downloads
```

### Workflow Summary
```
Mobile Photos + Public Datasets → Roboflow → Google Colab → T470s → RTSP
   (collect + merge)              (annotate)    (train)     (deploy)  (monitor)

Dataset Merging Workflow:
1. Collect custom data (mobile/CCTV)
2. Find 2-3 public datasets on Roboflow Universe
3. Upload custom images to Roboflow project
4. Annotate custom images
5. Import public datasets (Direct import method)
6. Map all classes to "corrosion"
7. Quality check (remove bad images, fix annotations)
8. Apply augmentation
9. Export and train!
```

---

## CONTACT RESOURCES

- Ultralytics Docs: https://docs.ultralytics.com
- Roboflow Learn: https://roboflow.com/learn
- Your PDF Methodology: Section 3.3-3.6
- HikVision SADP Tool: For finding camera IP

---

Last Updated: 2026