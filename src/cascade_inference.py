"""
Cascaded YOLOv12 Inference for Steel Corrosion Monitoring
Stage 1: Detection → Stage 2: Classification
Integrated with Real-Time Flask Telemetry Dashboard & Live UI Overlays
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import requests
import os
import subprocess
import sys
import threading
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Resolve the models directory relative to this file's location
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

class ThreadedCamera:
    def __init__(self, source):
        self.capture = cv2.VideoCapture(source)
        # Force OpenCV internal buffer to hold exactly 1 frame
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
        self.status, self.frame = self.capture.read()
        self.running = True
        
        # Start the background thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # This loop runs constantly in the background, grabbing the freshest frame
        while self.running:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(0.01) # Tiny sleep to prevent CPU overload

    def read(self):
        return self.status, self.frame
        
    def release(self):
        self.running = False
        self.thread.join()
        self.capture.release()
        
class CorrosionMonitor:
    def __init__(self, detection_model_path, classification_model_path):
        """
        Initialize cascaded corrosion monitoring system
        """
        print(" Loading models...")
        
        # Load Cloud Config
        load_dotenv()
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        if self.supabase_url and self.supabase_key:
            print(f"Cloud Sync Active: {self.supabase_url}")

        self.detector = YOLO(detection_model_path, task='detect')
        self.classifier = YOLO(classification_model_path, task='classify')

        # Detect Engine Type
        self.engine_name = "ONNX Runtime"
        if "openvino" in str(detection_model_path).lower() or str(detection_model_path).endswith('.xml'):
            self.engine_name = "OpenVINO"
        elif str(detection_model_path).endswith('.pt'):
            self.engine_name = "PyTorch"
        
        # Pure 0-3 grading scale
        self.grades = {
            0: "Healthy",
            1: "Mild",
            2: "Moderate",
            3: "Severe"
        }
        
        self.grade_colors = {
            0: (0, 255, 0),      # Green
            1: (0, 255, 255),    # Yellow
            2: (0, 165, 255),    # Orange
            3: (0, 0, 255)       # Red
        }

        self.fps_history = [] # Tracks FPS for smooth UI updates
        
        # Auto-screenshot setup for evidence logging
        self.screenshot_dir = RESULTS_DIR / "auto_screenshots"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.last_screenshot_time = 0  # Cooldown tracker (epoch seconds)
        self.screenshot_cooldown = 30  # Minimum seconds between auto-screenshots
        
        print(" Models loaded successfully!")
        print(f" Auto-screenshots saving to: {self.screenshot_dir}")
    
    def preprocess_image(self, image):
        """ Apply CLAHE contrast enhancement """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    def detect_corrosion(self, image, conf_threshold=0.45):
        """ Stage 1: Detect corrosion regions """
        results = self.detector.predict(image, conf=conf_threshold, iou=0.45, imgsz=640, verbose=False)
        return results[0]
    
    def classify_severity(self, roi_image):
        results = self.classifier.predict(roi_image, imgsz=224, verbose=False)
        probs = results[0].probs
        grade = probs.top1
        confidence = probs.top1conf.item() * 100 
            
        return grade, confidence

    def process_live_stream(self, camera_source=0, api_url="http://localhost:5000/api/update"):
        
        print(f" System Live. Monitoring camera source {camera_source}...")
        print(f" Broadcasting telemetry to {api_url}")

        # --- ACTIVATING THE THREADED CAMERA ---
        cap = ThreadedCamera(camera_source)

        # --- FORCED CINEMATIC FULLSCREEN MODE ---
        window_name = "Smart Corrosion Monitor - Master Inference"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            start_time = time.time() # Start stopwatch for FPS calculation
            
            # Now cap.read() will ALWAYS instantly return the freshest frame
            success, frame = cap.read()
            if not success or frame is None:
                continue # Skip to the next loop instead of crashing
            
            # --- AI PIPELINE ---
            enhanced_frame = self.preprocess_image(frame)
            detection_results = self.detect_corrosion(enhanced_frame)
            
            all_patches_in_frame = [] 
            detection_count = 0 

            if len(detection_results.boxes) > 0:
                for box in detection_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Min ROI size filter: skip patches smaller than 30x30px
                    roi_w, roi_h = x2 - x1, y2 - y1
                    if roi_w < 30 or roi_h < 30:
                        continue
                    
                    roi = frame[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        grade, conf = self.classify_severity(roi)
                        
                        # Save EVERY patch for the web dashboard
                        all_patches_in_frame.append({
                            "grade": f"Grade {grade}: {self.grades[grade]}",
                            "confidence": round(conf, 2),
                            "engine": self.engine_name
                        })

                        # Only draw bounding boxes for actual corrosion (Grade 1-3)
                        if grade == 0:
                            continue

                        # Draw thick, high-visibility bounding box and label
                        color = self.grade_colors[grade]
                        label = f"Grade-{grade}: {self.grades[grade]} ({conf:.1f}%)"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                        
                        # Filled background behind text for readability on live feed
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 4, y1 - 8), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                        
                        detection_count += 1

                # --- AUTO-SCREENSHOT: Save evidence when Grade 2+ detected ---
                has_severe = any(
                    "Moderate" in p["grade"] or "Severe" in p["grade"]
                    for p in all_patches_in_frame
                )
                if has_severe and (time.time() - self.last_screenshot_time) > self.screenshot_cooldown:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = self.screenshot_dir / f"alert_{timestamp}.jpg"
                    cv2.imwrite(str(screenshot_path), frame)
                    self.last_screenshot_time = time.time()
                    print(f"AUTO-SCREENSHOT: High-severity corrosion captured -> {screenshot_path.name}")

                # --- SEND DATA TO TELEMETRY CHANNELS ---
                if len(all_patches_in_frame) > 0:
                    payload = {"detections": all_patches_in_frame}
                    
                    # 1. Local Broadcast
                    try:
                        requests.post(api_url, json=payload, timeout=0.05)
                    except: pass

                    # 2. Cloud Broadcast (Supabase)
                    if self.supabase_url and self.supabase_key:
                        def broadcast_cloud():
                            headers = {
                                "apikey": self.supabase_key,
                                "Authorization": f"Bearer {self.supabase_key}",
                                "Content-Type": "application/json",
                                "Prefer": "return=minimal"
                            }
                            for detection in all_patches_in_frame:
                                data = {
                                    "grade": detection["grade"],
                                    "confidence": detection["confidence"],
                                    "engine": detection["engine"]
                                }
                                try:
                                    requests.post(f"{self.supabase_url}/rest/v1/detections", 
                                                 json=data, headers=headers, timeout=1.0)
                                except: pass
                        
                        # Threaded to prevent frame lag
                        threading.Thread(target=broadcast_cloud, daemon=True).start()
            
            else:
                # If no rust is found, display a clear status below the FPS
                cv2.putText(frame, "STATUS: CLEAR", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- CALCULATE & DRAW FPS ---
            elapsed_time = time.time() - start_time
            fps = 1 / max(elapsed_time, 0.001) # Max prevents division by zero
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            avg_fps = np.mean(self.fps_history)

            # Draw the beautiful overlay header
            cv2.putText(frame, f"FPS: {avg_fps:.1f} | Detections: {detection_count}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show the video feed (using the exact window_name variable!)
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Keep the FFMPEG lag-killers!
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|analyzeduration;1000000|probesize;1000000"

    monitor = CorrosionMonitor(
        detection_model_path=str(MODELS_DIR / "corrosion_detector_openvino_model"),
        classification_model_path=str(MODELS_DIR / "corrosion_grader_openvino_model")
    )
    
    # --- THE FULL-RES ASECAM CCTV STREAM ---
    cctv_url = "rtsp://admin:123456@192.168.1.163:554/user=admin&password=123456&channel=1&stream=0.sdp"

    # --- AUTO-LAUNCH API SERVER ---
    def start_api():
        api_path = ROOT_DIR / "src" / "api_server.py"
        subprocess.Popen([sys.executable, str(api_path)])
        print("Telemetry API started automatically...")

    # Start API in background if not already reachable
    try:
        requests.get("http://localhost:5000/api/live", timeout=0.5)
    except:
        start_api()
        time.sleep(1) # Give it a second to boot

    monitor.process_live_stream(camera_source=cctv_url)
