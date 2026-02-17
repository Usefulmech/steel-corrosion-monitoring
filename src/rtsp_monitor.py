"""
Real-time Corrosion Monitoring from RTSP CCTV Feed
For 8MP CCTV Camera (3.6mm focal length)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import os

class RTSPCorrosionMonitor:
    def __init__(self, detection_model_path, classification_model_path):
        print("🔧 Loading models...")
        self.detector = YOLO(detection_model_path)
        self.classifier = YOLO(classification_model_path)
        
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
        
        print("✅ Models loaded!")
    
    def preprocess_frame(self, frame):
        """Apply CLAHE enhancement for tropical conditions"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def process_frame(self, frame, frame_number):
        """Process single frame through cascaded pipeline"""
        start_time = time.time()
        
        # Enhance
        enhanced = self.preprocess_frame(frame)
        
        # Stage 1: Detect
        detection_results = self.detector.predict(
            enhanced,
            conf=0.5,
            verbose=False,
            imgsz=640  # Resize for faster processing
        )[0]
        
        annotated = frame.copy()
        detection_count = 0
        severity_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        high_severity_detections = []  # Track Grade 2 & 3 for alerts
        
        if len(detection_results.boxes) > 0:
            for idx, box in enumerate(detection_results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det_conf = box.conf.item()
                
                # Crop ROI
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0:
                    # Stage 2: Classify
                    cls_results = self.classifier.predict(roi, verbose=False)[0]
                    grade = cls_results.probs.top1
                    cls_conf = cls_results.probs.top1conf.item()
                    
                    severity_counts[grade] += 1
                    detection_count += 1
                    
                    # Track high severity for alerts
                    if grade >= 2:  # Moderate or Severe
                        high_severity_detections.append({
                            'grade': grade,
                            'confidence': cls_conf,
                            'bbox': (x1, y1, x2, y2)
                        })
                    
                    # Draw bounding box
                    color = self.grade_colors[grade]
                    thickness = 4 if grade >= 2 else 3  # Thicker for high severity
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                    
                    # Add label with background
                    label = f"Grade-{grade}: {self.grades[grade]} ({cls_conf:.2f})"
                    label_size, _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    cv2.rectangle(
                        annotated,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0] + 5, y1),
                        color,
                        -1
                    )
                    
                    cv2.putText(
                        annotated,
                        label,
                        (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
        
        # Calculate performance metrics
        processing_time = (time.time() - start_time) * 1000  # ms
        fps = 1 / (time.time() - start_time)
        
        # Create info panel
        panel_height = 150
        panel = np.zeros((panel_height, annotated.shape[1], 3), dtype=np.uint8)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            panel, timestamp, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
        )
        
        # FPS and processing time
        cv2.putText(
            panel,
            f"FPS: {fps:.1f} | Processing: {processing_time:.1f}ms | Frame: {frame_number}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        # Total detections
        cv2.putText(
            panel,
            f"Total Detections: {detection_count}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Severity breakdown
        y_pos = 100
        for grade in range(4):
            count = severity_counts[grade]
            color = self.grade_colors[grade]
            text = f"Grade-{grade} ({self.grades[grade]}): {count}"
            cv2.putText(
                panel, text, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            y_pos += 20
        
        # Alert indicator for high severity
        if high_severity_detections:
            alert_text = f"⚠ ALERT: {len(high_severity_detections)} High-Severity Detection(s)!"
            cv2.putText(
                panel, alert_text, (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        
        # Combine frame and panel
        result = np.vstack([annotated, panel])
        
        return result, high_severity_detections
    
    def run_rtsp(self, rtsp_url, save_output=False, output_dir="rtsp_output",
                 auto_screenshot_severity=2):
        """
        Run monitoring on RTSP stream
        
        Args:
            rtsp_url: RTSP URL (e.g., "rtsp://admin:password@ip:554/stream")
            save_output: Whether to save annotated video
            output_dir: Directory to save outputs
            auto_screenshot_severity: Auto-save screenshot if grade >= this (2=Moderate+)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/auto_screenshots", exist_ok=True)
        
        # Connect to RTSP stream
        print(f"📡 Connecting to RTSP stream...")
        print(f"   URL: {rtsp_url}")
        
        cap = cv2.VideoCapture(rtsp_url)
        
        # Optimize for low latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
        
        if not cap.isOpened():
            print("❌ Could not connect to RTSP stream!")
            print("\nTroubleshooting:")
            print("  1. Check RTSP URL format")
            print("  2. Verify camera is on network (ping IP)")
            print("  3. Confirm username/password")
            print("  4. Check port (usually 554)")
            print("  5. Ensure camera allows RTSP connections")
            return
        
        # Get stream properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Connected!")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print("\nControls:")
        print("   'q' - Quit")
        print("   's' - Save screenshot")
        print("   'r' - Start/stop recording")
        print(f"   Auto-screenshots for Grade-{auto_screenshot_severity}+ detections\n")
        
        # Video writer
        writer = None
        recording = save_output
        
        if recording:
            output_path = f"{output_dir}/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(
                output_path, fourcc, fps, (width, height + 150)
            )
            print(f"🔴 Recording to: {output_path}")
        
        frame_count = 0
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("⚠️ Lost connection. Attempting to reconnect...")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(rtsp_url)
                    if not cap.isOpened():
                        print("❌ Reconnection failed!")
                        break
                    continue
                
                # Process frame
                result_frame, high_severity = self.process_frame(frame, frame_count)
                
                # Auto-screenshot on high severity
                if high_severity and auto_screenshot_severity is not None:
                    for detection in high_severity:
                        if detection['grade'] >= auto_screenshot_severity:
                            screenshot_path = f"{output_dir}/auto_screenshots/severity{detection['grade']}_frame{frame_count}_{datetime.now().strftime('%H%M%S')}.jpg"
                            cv2.imwrite(screenshot_path, result_frame)
                            print(f"📸 Auto-screenshot: Grade-{detection['grade']} detected")
                            break  # One screenshot per frame
                
                # Save to video if recording
                if recording and writer is not None:
                    writer.write(result_frame)
                
                # Display
                cv2.imshow("RTSP Corrosion Monitor - Press 'q' to quit", result_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n⏹ Stopping monitoring...")
                    break
                    
                elif key == ord('s'):
                    screenshot_path = f"{output_dir}/manual_screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(screenshot_path, result_frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
                    screenshot_count += 1
                    
                elif key == ord('r'):
                    if not recording:
                        # Start recording
                        output_path = f"{output_dir}/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        writer = cv2.VideoWriter(
                            output_path, fourcc, fps, (width, height + 150)
                        )
                        recording = True
                        print(f"🔴 Recording started: {output_path}")
                    else:
                        # Stop recording
                        if writer is not None:
                            writer.release()
                            writer = None
                        recording = False
                        print("⏹ Recording stopped")
                
                frame_count += 1
                
                # Status update every 100 frames
                if frame_count % 100 == 0:
                    print(f"📊 Processed {frame_count} frames...")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"\n✅ Monitoring session complete!")
            print(f"   Total frames processed: {frame_count}")
            print(f"   Outputs saved to: {output_dir}")


if __name__ == "__main__":
    # Initialize monitor
    monitor = RTSPCorrosionMonitor(
        detection_model_path="models/detection_best.pt",
        classification_model_path="models/classification_best.pt"
    )
    
    # ===== CONFIGURE YOUR CAMERA =====
    # Replace with your camera's RTSP URL
    
    # HikVision Example (8MP):
    RTSP_URL = "rtsp://admin:your_password@192.168.1.64:554/Streaming/Channels/101"
    
    # If you don't know your RTSP URL:
    # 1. Use HikVision SADP tool to find camera IP
    # 2. Common HikVision format: rtsp://username:password@IP:554/Streaming/Channels/101
    # 3. Main stream (high quality): /Channels/101
    # 4. Sub stream (lower quality): /Channels/102
    
    # ===== RUN MONITORING =====
    monitor.run_rtsp(
        rtsp_url=RTSP_URL,
        save_output=False,  # Set True to auto-record
        output_dir="rtsp_monitoring_output",
        auto_screenshot_severity=2  # Auto-save when Grade 2+ detected
    )