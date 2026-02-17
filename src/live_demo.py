"""
Real-time CCTV/Webcam Corrosion Monitoring Demo
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

class LiveCorrosionMonitor:
    def __init__(self, detection_model_path, classification_model_path):
        self.detector = YOLO(detection_model_path)
        self.classifier = YOLO(classification_model_path)
        
        self.grades = {
            0: "Healthy",
            1: "Mild",
            2: "Moderate",
            3: "Severe"
        }
        
        self.grade_colors = {
            0: (0, 255, 0),
            1: (0, 255, 255),
            2: (0, 165, 255),
            3: (0, 0, 255)
        }
        
        self.fps_history = []
    
    def preprocess_frame(self, frame):
        """Apply CLAHE enhancement"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def process_frame(self, frame):
        """Process single frame through cascade"""
        start_time = time.time()
        
        # Enhance
        enhanced = self.preprocess_frame(frame)
        
        # Detect
        detection_results = self.detector.predict(
            enhanced,
            conf=0.5,
            verbose=False
        )[0]
        
        annotated = frame.copy()
        detection_count = 0
        
        if len(detection_results.boxes) > 0:
            for box in detection_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop and classify
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    cls_results = self.classifier.predict(roi, verbose=False)[0]
                    grade = cls_results.probs.top1
                    conf = cls_results.probs.top1conf.item()
                    
                    # Draw
                    color = self.grade_colors[grade]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"Grade-{grade}: {self.grades[grade]} ({conf:.2f})"
                    cv2.putText(
                        annotated,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
                    detection_count += 1
        
        # Calculate FPS
        fps = 1 / (time.time() - start_time)
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = np.mean(self.fps_history)
        
        # Add FPS overlay
        cv2.putText(
            annotated,
            f"FPS: {avg_fps:.1f} | Detections: {detection_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return annotated
    
    def run(self, camera_index=0):
        """Run live demo"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("❌ Could not open camera!")
            return
        
        print("✅ Camera opened. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to grab frame")
                break
            
            # Process
            result_frame = self.process_frame(frame)
            
            # Display
            cv2.imshow("Corrosion Monitor - Live Demo", result_frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Demo stopped.")


if __name__ == "__main__":
    monitor = LiveCorrosionMonitor(
        detection_model_path="models/detection_best.pt",
        classification_model_path="models/classification_best.pt"
    )
    
    # Use 0 for default webcam, or specify camera index
    monitor.run(camera_index=0)