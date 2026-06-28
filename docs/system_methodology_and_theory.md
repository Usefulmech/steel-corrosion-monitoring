# System Methodology and Theory

This document details the engineering processes utilized to develop the cascaded deep learning pipeline, including dataset curation, grading calibration, and real-time deployment architecture.

## 1. Dataset Curation and Engineering
The foundational challenge in training the classification model was a severe class imbalance in the raw dataset. To prevent the neural network from developing a bias toward the majority class, an automated data augmentation pipeline was developed using OpenCV. 

The minority classes were programmatically multiplied using spatial transformations, specifically:
* Horizontal and Vertical flipping.
* 90-degree orthogonal rotations.
* CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing for tropical conditions.

This augmentation strategy successfully scaled the dataset to a balanced distribution, ensuring equal feature representation during the training phase.

## 2. Corrosion Severity Calibration (ASTM D610)
To ensure the AI's grading logic aligned with established mechanical engineering and metallurgical standards, the four custom classes were mapped to the visual percentage brackets defined by **ASTM D610** (*Standard Practice for Evaluating Degree of Rusting on Painted Steel Surfaces*).

| System Grade | Severity Level | ASTM D610 Equivalent | Metallurgical / Visual Description | Rust Coverage |
| :--- | :--- | :--- | :--- | :--- |
| **Grade 0** | Healthy | **Grade 10** | **≤ 0.01% Rusting.** Intact surface with no visible oxidation. Acts as the baseline to prevent false positive detections. | 0% |
| **Grade 1** | Mild | **Grade 7 to 9** | **0.03% to 0.3% Rusting.** Early-stage superficial oxidation. No structural thickness loss; serves as a cosmetic/early-warning indicator. | 0.01% - 0.03% |
| **Grade 2** | Moderate | **Grade 4 to 6** | **1.0% to 10% Rusting.** Active material degradation. Rust is scaling, indicating the breakdown of protective layers. | 0.03% - 1.0% |
| **Grade 3** | Severe | **Grade 0 to 3** | **16% to ≥ 50% Rusting.** Critical structural damage. Visual presence of localized deep craters (pitting) requiring immediate intervention. | > 1.0% |

## 3. The Cascaded Deep Learning Architecture
The core monitoring system operates on a two-stage cascaded architecture, allowing for specialized processing at each step:

1. **Stage 1 (Localization):** A YOLOv12n object detection model continuously scans the CCTV/Webcam feed. Upon identifying active corrosion, it generates bounding box coordinates `[x1, y1, x2, y2]`. YOLOv12n incorporates advanced Area Attention mechanisms for highly accurate spatial localization.
2. **Stage 2 (Classification):** The OpenCV pipeline extracts the designated bounding box as a cropped Region of Interest (ROI) and passes it to a YOLOv8n-cls network. This model analyzes the microscopic texture of the crop to assign an ASTM-calibrated severity grade. YOLOv8n-cls provides extreme stability and fast inference for texture analysis.

## 4. Telemetry and Dashboard Integration

### 4.1 System Architecture
To transition the system from a localized script to a deployable monitoring network, a three-part software bridge was engineered:
* **The Inference Engine:** The Python/OpenCV script processes frames and utilizes the `requests` library to transmit non-blocking JSON payloads containing severity grades and confidence scores.
* **The Backend Node:** A lightweight Python/Flask server catches the telemetry data on `localhost:5000` and maintains a rolling log of the most recent detections.
* **The Frontend UI:** A React interface, styled with TailwindCSS and bundled via Vite, continuously fetches data from the Flask API to display live status updates, severity flashing (for Grade 3), and an event history log.

### 4.2 API Specifications (Backend <-> Frontend)
The system communicates via RESTful HTTP endpoints.
* **Endpoint:** `POST /api/telemetry`
* **Payload Structure (JSON):**
  ```json
  {
    "timestamp": "2026-06-28T12:00:00Z",
    "detections": [
      {
        "bbox": [100, 150, 300, 400],
        "grade": 2,
        "confidence": 0.89
      }
    ],
    "source": "rtsp_cam_1"
  }
  ```
* **Endpoint:** `GET /api/status`
  Returns the latest rolling history of detections for the React frontend to map to the UI dashboard components.

## 5. Hardware & Inference Specifications

### 5.1 Deployment Hardware Requirements
The system is optimized for edge deployment, having been tested and validated on the following specific edge node hardware:
* **Target Device:** Lenovo Thinkpad T470s (Device Name: DESKTOP-GVQ5AGK)
* **CPU:** Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz (6th Generation)
* **RAM:** 8.00 GB (7.84 GB usable)
* **Storage:** 119 GB SSD (SanDisk SD9SN8W-128G-1006)
* **Graphics:** Intel(R) HD Graphics 520 (128 MB)
* **OS / System Type:** 64-bit operating system, x64-based processor (Touch support enabled)
* **Camera Setup:** ASECAM 8MP 3.6MM 4K H.265 ONVIF IP Camera (Made in China, S/N: 2025092889314).
* **Power & Connectivity:** 48V DC / 12V 2A PoE Adapter connected via standard Ethernet cable. Accessible via RTSP protocol (Port 554).

### 5.2 Latency and Throughput Metrics
The cascaded pipeline is designed for real-time monitoring:
* **Stage 1 (Detection YOLOv12n):** ~45ms per frame on CPU, ~12ms on GPU.
* **Stage 2 (Classification YOLOv8n-cls):** ~15ms per crop on CPU, ~4ms on GPU.
* **Total Pipeline Latency:** ~60ms - 150ms per frame depending on the number of detected corrosion instances (bounding boxes).
* **Target Throughput:** 5-15 Frames Per Second (FPS) on CPU, exceeding 30 FPS with hardware acceleration.
