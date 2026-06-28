# Training Metrics Report
## Development of a Smart Corrosion Monitoring System for Steel Structures Using CCTV

### Trained on: Kaggle (Dual Tesla T4 GPUs, 15GB VRAM each)

---

## 1. Detection Model (YOLOv12n)

### Model Architecture
- **Base Model:** YOLOv12n (custom download from official repo)
- **Backbone:** R-ELAN with Area Attention (A2C2f modules)
- **Task:** Object Detection (single class: "corrosion")
- **Parameters:** 2,568,243 (2,556,923 fused)
- **GFLOPs:** 6.5 (6.3 fused)
- **Layers:** 272 (159 fused)
- **Model Size:** 5.5 MB (.pt)

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Image Size | 640x640 |
| Batch Size | 16 |
| Optimizer | SGD |
| Learning Rate (initial) | 0.01 |
| Learning Rate (final) | 0.01 (cosine annealing) |
| Momentum | 0.937 |
| Weight Decay | 0.0005 |
| Early Stopping Patience | 50 |
| Device | GPU 0 (Tesla T4) |

### Augmentation Pipeline
| Augmentation | Value |
|-------------|-------|
| HSV Hue | 0.015 |
| HSV Saturation | 0.7 |
| HSV Value | 0.4 |
| Rotation | 15 degrees |
| Horizontal Flip | 0.5 |
| Vertical Flip | 0.0 |
| Mosaic | 1.0 |
| Erasing | 0.4 |
| Albumentations | Blur, MedianBlur, ToGray, CLAHE |

### Dataset
- **Training Images:** 2,262 (60 background images)
- **Validation Images:** 173
- **Total Instances (val):** 519 bounding boxes
- **Classes:** 1 (corrosion)
- **Source:** Roboflow (usefulmech/corrosion-detection-cckc1, version 1)
- **Format:** YOLOv8

### Final Validation Metrics (Best Model)
| Metric | Value |
|--------|-------|
| **Precision (P)** | **0.699** |
| **Recall (R)** | **0.443** |
| **mAP@50** | **0.498** |
| **mAP@50-95** | **0.325** |

### Inference Speed (Tesla T4)
| Stage | Time |
|-------|------|
| Preprocess | 0.2 ms |
| Inference | 4.0 ms |
| Loss | 0.0 ms |
| Postprocess | 4.9 ms |
| **Total** | **~9.1 ms/image** |

### Training Progress (Key Epochs)
| Epoch | Box Loss | Cls Loss | DFL Loss | P | R | mAP@50 | mAP@50-95 |
|-------|----------|----------|----------|------|------|--------|-----------|
| 1 | 1.573 | 2.352 | 1.825 | 0.439 | 0.210 | 0.202 | 0.113 |
| 50 | 1.209 | 1.597 | 1.603 | 0.609 | 0.310 | 0.350 | 0.205 |
| 100 | 1.080 | 1.342 | 1.482 | 0.689 | 0.401 | 0.447 | 0.286 |
| 145 | 0.992 | 1.190 | 1.410 | 0.748 | 0.395 | 0.475 | 0.307 |
| 160 | 0.961 | 1.127 | 1.380 | 0.675 | 0.429 | 0.488 | 0.314 |
| 185 | 0.932 | 1.062 | 1.351 | 0.716 | 0.418 | 0.487 | 0.314 |
| 198 | 0.874 | 0.894 | 1.334 | 0.699 | 0.443 | 0.498 | 0.325 |
| 200 | 0.878 | 0.894 | 1.345 | 0.690 | 0.438 | 0.493 | 0.326 |

### Training Duration
- **Total Time:** 2.007 hours (200 epochs)
- **Average Speed:** ~4.1 it/s (142 batches per epoch)

---

## 2. Classification Model (YOLOv8n-cls)

### Model Architecture
- **Base Model:** YOLOv8n-cls (pretrained on ImageNet)
- **Task:** Image Classification (4 classes)
- **Parameters:** 1,443,412 (1,440,004 fused)
- **GFLOPs:** 3.4 (3.3 fused)
- **Layers:** 56 (30 fused)
- **Model Size:** 3.0 MB (.pt)

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Image Size | 224x224 |
| Batch Size | 32 |
| Optimizer | AdamW (auto-selected) |
| Learning Rate | 0.00125 (auto-tuned) |
| Momentum | 0.9 |
| Weight Decay | 0.0005 |
| Device | GPU 0 (Tesla T4) |

### Dataset
- **Training Images:** 1,313
- **Validation Images:** 330
- **Total Images:** 1,643
- **Classes:** 4 (Grade 0, Grade 1, Grade 2, Grade 3)
- **Split Ratio:** 80% Train / 20% Validation
- **Grade 0 Source:** 149 original clean steel patches augmented to 415

### Class Distribution
| Grade | Label | Approx Count |
|-------|-------|---------------|
| 0 | Healthy | 415 images |
| 1 | Mild | ~400 images |
| 2 | Moderate | ~400 images |
| 3 | Severe | ~428 images |

### Final Validation Metrics (Best Model)
| Metric | Value |
|--------|-------|
| **Top-1 Accuracy** | **86.4%** |
| **Top-5 Accuracy** | **100.0%** |

### Inference Speed (Tesla T4)
| Stage | Time |
|-------|------|
| Preprocess | 0.3 ms |
| Inference | 0.6 ms |
| Loss | 0.0 ms |
| Postprocess | 0.0 ms |
| **Total** | **~0.9 ms/image** |

### Training Progress (Key Epochs)
| Epoch | Loss | Top-1 Acc | Top-5 Acc |
|-------|------|-----------|-----------|
| 1 | 1.2400 | 71.5% | 100% |
| 7 | 0.5190 | 83.9% | 100% |
| 13 | 0.3355 | 84.8% | 100% |
| 23 | 0.2576 | 85.2% | 100% |
| 32 | 0.1920 | 86.4% | 100% |
| 34 | 0.1331 | 85.8% | 100% |
| 42 | 0.1121 | 85.5% | 100% |
| 50 | 0.1132 | 84.8% | 100% |

### Training Duration
- **Total Time:** 0.060 hours (~3.6 minutes)
- **Average Speed:** ~12 it/s (42 batches per epoch)

---

## 3. Combined Pipeline Performance

### Cascaded Inference (Detection + Classification)
| Component | Engine | Speed per Frame |
|-----------|--------|-----------------|
| CLAHE Preprocessing | OpenCV | ~2 ms |
| YOLOv12n Detection | OpenVINO | ~5 ms |
| YOLOv8n Classification (per ROI) | OpenVINO | ~1 ms |
| Overlay Drawing | OpenCV | ~1 ms |
| **Total (1 ROI)** | | **~9 ms** |

### Production Thresholds
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Detection Confidence | 0.45 | Filters weak false positives without losing real detections |
| IoU (NMS) | 0.45 | Merges overlapping boxes on heavily corroded areas |
| Min ROI Size | 30x30 px | Rejects noise-level patches too small for meaningful grading |
| Grade-0 Filter | Enabled | Suppresses bounding box overlay on healthy steel patches |
| Auto-Screenshot | Grade 2+ | Evidence capture with 30-second cooldown |

### Deployment Specifications
| Spec | Value |
|------|-------|
| Target Hardware | Lenovo T470s (Intel i5-7300U, 8GB RAM) |
| Primary Engine | OpenVINO (FP32) |
| Fallback Engine | ONNX Runtime / PyTorch |
| CCTV Camera | 8MP ASECAM (RTSP protocol) |
| Expected Live FPS | 4-8 FPS (depends on detection count) |
