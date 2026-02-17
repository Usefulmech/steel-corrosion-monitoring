# Smart Steel Corrosion Monitoring System Using CCTV

A cascaded YOLOv12 + YOLOv8 deep learning system for real-time detection and severity grading of uniform steel corrosion using CCTV footage.

![System Architecture](docs/images/architecture_diagram.png)

## 🎓 Project Information

**Author:** Adeniji Yusuf Joseph (Matric: 20201777)  
**Institution:** Federal University of Agriculture, Abeokuta  
**Department:** Mechanical Engineering  
**Supervisor:** Prof. O. R. Adetunji  
**Year:** 2026

## 📋 Abstract

This project develops an automated corrosion monitoring system using a cascaded deep learning architecture combining YOLOv12 for detection and YOLOv8 for classification. The system processes CCTV footage to detect uniform corrosion and grade severity (Grade 0-3) according to ASTM D610 standards, enabling proactive maintenance of Nigeria's industrial infrastructure.

## 🌟 Key Features

- ✅ **Real-time Detection:** YOLOv12n for corrosion localization
- ✅ **Fine-grained Grading:** YOLOv8n-cls for 4-class severity classification (Grade 0-3)
- ✅ **CLAHE Preprocessing:** Enhanced performance in tropical conditions
- ✅ **RTSP Integration:** Live monitoring from 8MP CCTV cameras
- ✅ **Edge Deployment:** Optimized for Lenovo T470s (8GB RAM)
- ✅ **Auto-screenshots:** Automatic capture of high-severity detections

## 🏗️ System Architecture

### Cascaded Two-Stage Pipeline:

1. **Stage 1 - Detection (YOLOv12n):**
   - Localizes all corrosion regions
   - R-ELAN backbone for stability
   - Area Attention module for feature enhancement

2. **Stage 2 - Classification (YOLOv8n-cls):**
   - Grades severity: Grade 0 (Healthy) to Grade 3 (Severe)
   - Based on ASTM D610 rust area percentage standards
   - Resolves inter-class ambiguity

## 📊 Dataset

- **Total Images:** 2,500+ images
- **Sources:**
  - Custom data: 600 images (mobile + CCTV)
  - Public datasets: 1,900 images (Roboflow Universe)
- **Classes:** 4 severity grades (0-3)
- **Augmentation:** CLAHE, rotation, brightness, HSV, mosaic

**Dataset Links:**
- Detection: [Roboflow Project Link]
- Classification: [Roboflow Project Link]

## 🎯 Performance Metrics

### Detection Model (YOLOv12n)
- mAP50: 0.XX
- Precision: 0.XX
- Recall: 0.XX
- Inference: XX ms/image

### Classification Model (YOLOv8n-cls)
- Top-1 Accuracy: 0.XX
- F1-Score: 0.XX
- Confusion Matrix: [Link to image]

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/steel-corrosion-monitoring.git
cd steel-corrosion-monitoring

# Create virtual environment
python -m venv corrosion_env
corrosion_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download model weights
python scripts/download_models.py
```

### Usage

**1. Test on Single Image:**
```bash
python src/cascade_inference.py --image path/to/image.jpg
```

**2. RTSP Live Monitoring:**
```bash
python src/rtsp_monitor.py --rtsp-url "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101"
```

**3. Batch Processing:**
```bash
python scripts/evaluate_model.py --input-dir path/to/images --output-dir results/
```

## 📖 Documentation

- [Implementation Guide](docs/IMPLEMENTATION_GUIDE.md) - Complete step-by-step guide
- [Quick Reference](docs/QUICK_REFERENCE.md) - Cheat sheet
- [Methodology](docs/METHODOLOGY.md) - Research methodology
- [Results](docs/RESULTS.md) - Experimental results and analysis

## 🛠️ Technology Stack

- **Deep Learning:** Ultralytics YOLOv12, YOLOv8
- **Computer Vision:** OpenCV, PIL
- **Training:** Google Colab (free GPU)
- **Annotation:** Roboflow
- **Deployment:** Python 3.8+, Edge computing (T470s)
- **Camera:** 8MP CCTV (3.6mm focal length, RTSP)

## 📁 Project Structure
steel-corrosion-monitoring/
├── docs/              # Documentation
├── src/               # Source code
├── scripts/           # Utility scripts
├── notebooks/         # Jupyter notebooks
├── configs/           # Configuration files
├── models/            # Model weights (download separately)
├── data/              # Dataset information
└── results/           # Experimental results
## 🎓 Academic Context

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

## 📄 Citation

If you use this work, please cite:
```bibtex
@thesis{adeniji2026corrosion,
  author = {Adeniji Yusuf Joseph},
  title = {Development of a Smart Steel Corrosion Monitoring System Using Closed Circuit Television (CCTV)},
  school = {Federal University of Agriculture, Abeokuta},
  year = {2026},
  type = {Bachelor's Thesis},
  department = {Mechanical Engineering}
}
```

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Prof. O. R. Adetunji (Supervisor)
- Department of Mechanical Engineering, FUNAAB
- Roboflow Community for public datasets
- Ultralytics for YOLO frameworks

## 📧 Contact

**Adeniji Yusuf Joseph**  
Email: your.email@example.com  
LinkedIn: [Your LinkedIn]  
GitHub: [@yourusername](https://github.com/yourusername)

---

**Status:** 🚧 Active Development | Last Updated: February 2026