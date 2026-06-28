# Chapter 4: Results and Model Evaluation

This section evaluates the performance metrics of the trained models and the efficiency of the real-time cascaded deployment pipeline.

## 4.1 Training Configuration and Data Splitting
The augmented dataset was partitioned using an 80/20 split methodology. 
* **Training Set (80%):** Utilized strictly for feature extraction and weight adjustments.
* **Validation Set (20%):** Kept completely unseen by the model during the training phase to serve as an honest testing ground for generalization.

The Stage 2 severity classifier (`yolov8n-cls.pt`) was trained over **50 epochs** with a batch size of 32 and an image resolution of 224x224 pixels. The training was hardware-accelerated using dual T4 GPUs.

## 4.2 Classification Performance Metrics
The primary metric for evaluating the YOLOv8 grading model is the Top-1 Accuracy, which measures how often the model's absolute first choice matches the actual ground-truth label of the unseen validation data.

* **Top-1 Accuracy:** `0.864` (86.4%)
* **Top-5 Accuracy:** `1.000` (100%)

Achieving an **86.4% Top-1 Accuracy** on the isolated 20% validation set demonstrates robust feature learning. It indicates that the neural network successfully modeled the complex, highly subjective morphological differences between uniform scaling (Grade 2) and localized pitting (Grade 3), rather than simply memorizing the training batch. 

*(Note regarding architectural metrics: Because this specific classification environment contains only four distinct classes (Grades 0-3), the Top-5 accuracy is mathematically guaranteed to be 100% and is recorded strictly as an artifact of the YOLO evaluation architecture).*

## 4.3 Real-Time Inference Performance
During live deployment via the `cascade_inference.py` pipeline, the system successfully managed the dual-model workload while maintaining a stable frame rate.

* **Image Enhancement:** CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing consistently normalized lighting conditions without introducing latency bottlenecks.
* **Network Handoff:** The inclusion of a 50-millisecond timeout (`timeout=0.05`) in the API broadcasting logic ensured that the camera feed's FPS remained smooth and uninterrupted, even during high-frequency network transmission to the Flask telemetry server. 
* **Simultaneous Tracking:** The algorithm accurately captured and transmitted data for multiple independent corrosion patches within a single frame, successfully logging all instances to the React dashboard event table.