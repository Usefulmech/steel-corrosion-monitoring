# Development of a Smart Corrosion Monitoring System for Steel Structures Using CCTV

A Cascaded YOLOv12 and YOLOv8 Deep Learning System for Real-Time Detection and Severity Grading of Steel Corrosion using CCTV Footage.

## Project Information

**Author:** Adeniji Yusuf Joseph (Matric: 20201777)  
**Institution:** Federal University of Agriculture, Abeokuta  
**Department:** Mechanical Engineering  
**Supervisor:** Prof. O. R. Adetunji  
**Year:** 2026

## Abstract

This project develops an automated corrosion monitoring system using a cascaded deep learning architecture combining YOLOv12 for detection and YOLOv8 for classification. The system processes CCTV footage to detect uniform corrosion and grade severity (Grade 0-3) according to ASTM D610 standards, enabling proactive maintenance of Nigeria's industrial infrastructure.

## Performance Metrics

### Detection Model (YOLOv12n)
| Metric | Value |
|--------|-------|
| Precision | 0.699 |
| Recall | 0.443 |
| mAP@50 | 0.498 |
| mAP@50-95 | 0.325 |
| Inference | 4.0 ms/image (T4 GPU) |

### Classification Model (YOLOv8n-cls)
| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 86.4% |
| Top-5 Accuracy | 100.0% |
| Inference | 0.6 ms/image (T4 GPU) |

*Full training metrics, loss curves, and hyperparameter details: [training_metrics.md](results/metrics/training_metrics.md)*

## Key Features

- **Real-time Detection:** YOLOv12n for corrosion localization
- **Fine-grained Grading:** YOLOv8n-cls for 4-class severity classification (Grade 1-3 visualized, Grade 0 filtered)
- **CLAHE Preprocessing:** Enhanced performance in tropical lighting conditions
- **RTSP Integration:** Live monitoring from 8MP CCTV cameras via threaded capture
- **Edge Deployment:** Optimized for Lenovo T470s (8GB RAM) with OpenVINO acceleration
- **Auto-screenshots:** Automatic evidence capture when Grade 2+ (Moderate/Severe) corrosion is detected
- **Telemetry Dashboard:** Real-time React + Flask web interface with Supabase cloud sync
- **Dual Engine Support:** Both OpenVINO (optimized) and PyTorch (baseline) inference paths

## System Architecture

### Cascaded Two-Stage AI Pipeline

1. **Stage 1 - Detection (YOLOv12n):**
   - Localizes all corrosion regions in the frame
   - R-ELAN backbone for training stability
   - Area Attention module for feature enhancement
   - Optimized thresholds: `conf=0.45`, `iou=0.45`

2. **Stage 2 - Classification (YOLOv8n-cls):**
   - Grades severity: Grade 0 (Healthy) to Grade 3 (Severe)
   - Based on ASTM D610 rust area percentage standards
   - Grade-0 detections are filtered from visualization (false positive suppression)
   - Minimum ROI filter: patches smaller than 30x30px are skipped

3. **Stage 3 - Dashboard (React + Flask):**
   - Real-time visualization of corrosion severity
   - Event log of all detections with timestamps
   - Cloud sync via Supabase for remote supervisor access
   - Auto-screenshot evidence logging for Grade 2+ detections

### Inference Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `conf` | 0.45 | Minimum detection confidence (eliminates false positives on concrete, dirt, etc.) |
| `iou` | 0.45 | NMS overlap threshold (merges duplicate bounding boxes) |
| `imgsz` (detection) | 640 | Input resolution for YOLOv12n detector |
| `imgsz` (classification) | 224 | Input resolution for YOLOv8n classifier |
| CLAHE `clipLimit` | 2.0 | Contrast enhancement intensity |
| Min ROI size | 30x30px | Minimum patch size for classification |
| Screenshot cooldown | 30s | Minimum interval between auto-screenshots |

## Dataset

- **Total Images:** 2,500+ images
- **Sources:**
  - Custom data: 600 images (mobile + CCTV)
  - Public datasets: 1,900 images (Roboflow Universe)
- **Classes:** 4 severity grades (0-3)
- **Augmentation:** CLAHE, rotation, brightness, HSV, mosaic

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/Usefulmech/steel-corrosion-monitoring.git
cd steel-corrosion-monitoring

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirement.txt
```

### Usage

**1. Test on a Single Image (OpenVINO optimized):**
```bash
python src/test_single_image.py
```

**2. Test on a Single Image (PyTorch baseline):**
```bash
python src/test_single_image_baseline.py
```

**3. Live CCTV Monitoring (OpenVINO optimized):**
```bash
python src/cascade_inference.py
```

**4. Live CCTV Monitoring (PyTorch/ONNX baseline):**
```bash
python src/cascade_inference_baseline.py
```

**5. Export Models to OpenVINO/ONNX:**
```bash
python scripts/export_lossless.py
```

**6. Start Web Dashboard (Backend & Frontend):**

*Start the Flask Backend Server (Port 5000):*
```bash
python src/api_server.py
```

*Start the React Frontend (Port 5173):*
```bash
cd frontend
npm install
npm run dev
```

**7. Local Network Access (View on Mobile):**

*To access the dashboard from your phone or another device on the same Wi-Fi network, find your computer's local IP address (e.g., `192.168.1.100`) and use that instead of `localhost`.*

*Both the frontend (`vite --host`) and backend (`host='0.0.0.0'`) are already configured to allow local network connections.*

```bash
# Example URL to enter in your phone's browser:
http://<YOUR_LOCAL_IP>:5173
```
*Note: Ensure your computer's firewall allows inbound connections on ports 5000 and 5173.*

## Project Structure

```
steel-corrosion-monitoring/
├── src/                          # Core inference source code
│   ├── cascade_inference.py      #   Live CCTV pipeline (OpenVINO)
│   ├── cascade_inference_baseline.py  #   Live CCTV pipeline (PyTorch/ONNX)
│   ├── test_single_image.py      #   Static image tester (OpenVINO)
│   ├── test_single_image_baseline.py  #   Static image tester (PyTorch)
│   └── api_server.py            #   Flask telemetry API for dashboard
│
├── models/                       # Trained model weights
│   ├── corrosion_detector.pt     #   YOLOv12n detector (PyTorch)
│   ├── corrosion_grader.pt       #   YOLOv8n classifier (PyTorch)
│   ├── corrosion_detector.onnx   #   YOLOv12n detector (ONNX FP32)
│   ├── corrosion_grader.onnx     #   YOLOv8n classifier (ONNX FP32)
│   ├── corrosion_detector_openvino_model/  #   Detector (OpenVINO IR)
│   └── corrosion_grader_openvino_model/    #   Classifier (OpenVINO IR)
│
├── results/                      # All inference outputs
│   ├── sample_input/             #   Raw test images (unprocessed)
│   ├── sample_output/            #   Annotated inference results
│   ├── auto_screenshots/         #   Auto-captured high-severity evidence
│   └── metrics/                  #   Training metrics and graphs
│
├── scripts/                      # Utility scripts
│   ├── export_lossless.py        #   Export PyTorch -> ONNX + OpenVINO (FP32)
│   ├── export_openvino.py        #   Export PyTorch -> OpenVINO (FP16)
│   └── extract_frames.py         #   Extract frames from video/RTSP for annotation
│
├── notebook/                     # Training notebooks (Kaggle)
│   ├── yolov12n_corrosion_detection_training.ipynb   # Detection training
│   └── yolov8_corrosion_classification_training.ipynb # Classification training
│
├── frontend/                     # React dashboard (Vite)
│
├── docs/                         # Technical documentation
│   ├── implementation_guide.md   #   Full implementation walkthrough
│   ├── methodology.md            #   Research methodology
│   ├── results.md                #   Experimental results
│   ├── quick-reference.md        #   Quick reference cheat sheet
│   ├── writeup_reference.md      #   Thesis writing reference
│   └── annotation_guideline.md   #   Dataset annotation guide
│
├── .env                          # Supabase cloud credentials
├── README.md                     # This file
├── README_CLOUD.md               # Cloud deployment guide (Supabase + Vercel)
├── requirement.txt               # Python dependencies
└── LICENSE                       # MIT License
```

## Technology Stack

- **Deep Learning:** Ultralytics YOLOv12, YOLOv8
- **Inference Engines:** OpenVINO (optimized), ONNX Runtime (baseline), PyTorch (baseline)
- **Computer Vision:** OpenCV (CLAHE preprocessing, annotation rendering)
- **Training Platform:** Kaggle (Dual T4 GPU)
- **Dataset Management:** Roboflow
- **Dashboard:** React (Vite) + Flask + Supabase (real-time cloud sync)
- **Deployment:** Python 3.8+, Edge computing on Lenovo T470s (Intel i5, 8GB RAM)
- **Camera:** 8MP ASECAM CCTV (3.6mm focal length, RTSP protocol)

## Academic Context

This project addresses:
- **SDG 9:** Industry, Innovation, and Infrastructure
- **SDG 11:** Sustainable Cities and Communities
- **SDG 12:** Responsible Consumption and Production

**Research Gap:** Lack of automated, fine-grained severity grading systems for uniform corrosion in Nigerian infrastructure.

**Novel Contributions:**
1. Cascaded architecture for inter-class ambiguity resolution
2. CLAHE preprocessing pipeline for tropical environments
3. Edge-deployable system for remote monitoring
4. ASTM D610-aligned objective grading
5. Deterministic 0-3 grading scale (no uncertainty buffer)

## Citation

If you use this work, please cite:
```bibtex
@thesis{adeniji2026corrosion,
  author = {Adeniji Yusuf Joseph},
  title = {Development of a Smart Corrosion Monitoring System for Steel Structures Using CCTV},
  school = {Federal University of Agriculture, Abeokuta},
  year = {2026},
  type = {Bachelor's Thesis},
  department = {Mechanical Engineering}
}
```

## Future Research Opportunities

Building upon the successful implementation of this smart corrosion monitoring system, several avenues for future research and enhancement exist:

1. **Integration with Drone/UAV Technology**: While currently optimized for fixed CCTV, adapting the models to process live feeds from inspection drones would allow for the monitoring of hard-to-reach steel structures like bridges, offshore rigs, and high-rise scaffolds.
2. **Expansion of Defect Classes**: Future work could expand the classification taxonomy to detect other structural anomalies beyond corrosion, such as micro-cracks, pitting, weld defects, and coating failures.
3. **Multi-modal Sensor Fusion**: Combining the computer vision system with other non-destructive testing (NDT) sensors (e.g., ultrasonic thickness gauges, thermal cameras) could yield a more robust predictive maintenance framework by correlating visual surface degradation with structural depth loss.
4. **Edge Computing Optimization**: Further quantization of the YOLOv12n and YOLOv8-cls models could enable deployment on ultra-low-power edge devices (like Raspberry Pi or specialized NPUs) directly on-site, removing the need for continuous cloud connectivity.
5. **Longitudinal Degradation Tracking**: Implementing a temporal tracking module that compares the same structural region across months or years to calculate the exact rate of corrosion spread, aiding in predictive life-cycle modeling of the steel structures.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Prof. O. R. Adetunji (Supervisor)
- Department of Mechanical Engineering, FUNAAB
- Roboflow Community for public datasets
- Ultralytics for YOLO frameworks

## Contact

**Adeniji Yusuf Joseph**  
Email: [yusufadeniji23@gmail.com](mailto:yusufadeniji23@gmail.com)  
GitHub: [@Usefulmech](https://github.com/Usefulmech)

---

**Status:** Production Ready | Last Updated: June 2026