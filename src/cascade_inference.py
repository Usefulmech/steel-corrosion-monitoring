"""
Cascaded YOLOv12 Inference for Steel Corrosion Monitoring
Stage 1: Detection → Stage 2: Classification
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import time

class CorrosionMonitor:
    def __init__(self, detection_model_path, classification_model_path):
        """
        Initialize cascaded corrosion monitoring system
        
        Args:
            detection_model_path: Path to YOLOv12n detection model (.pt file)
            classification_model_path: Path to YOLOv8n-cls classification model (.pt file)
        """
        print("🔧 Loading models...")
        print("   Detection: YOLOv12n")
        print("   Classification: YOLOv8n-cls")
        
        self.detector = YOLO(detection_model_path)
        self.classifier = YOLO(classification_model_path)
        
        # Severity grades
        self.grades = {
            0: "Grade-0: Healthy",
            1: "Grade-1: Mild",
            2: "Grade-2: Moderate",
            3: "Grade-3: Severe"
        }
        
        # Grade colors (BGR format for OpenCV)
        self.grade_colors = {
            0: (0, 255, 0),      # Green
            1: (0, 255, 255),    # Yellow
            2: (0, 165, 255),    # Orange
            3: (0, 0, 255)       # Red
        }
        
        print("✅ Models loaded successfully!")
    
    def preprocess_image(self, image):
        """
        Apply CLAHE contrast enhancement
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge and convert back to BGR
        lab_clahe = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_corrosion(self, image, conf_threshold=0.5):
        """
        Stage 1: Detect corrosion regions
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold
        
        Returns:
            Detection results
        """
        results = self.detector.predict(
            image,
            conf=conf_threshold,
            verbose=False
        )
        return results[0]
    
    def classify_severity(self, roi_image):
        """
        Stage 2: Classify corrosion severity
        
        Args:
            roi_image: Cropped region of interest
        
        Returns:
            Predicted grade (0-3) and confidence
        """
        results = self.classifier.predict(
            roi_image,
            verbose=False
        )
        
        # Get top prediction
        probs = results[0].probs
        grade = probs.top1
        confidence = probs.top1conf.item()
        
        return grade, confidence
    
    def process_image(self, image_path, output_dir="output", save_crops=True):
        """
        Complete cascaded processing pipeline
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            save_crops: Whether to save cropped ROIs
        
        Returns:
            Dictionary with results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        if save_crops:
            os.makedirs(f"{output_dir}/crops", exist_ok=True)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_image = image.copy()
        
        # Preprocessing
        enhanced = self.preprocess_image(image)
        
        # Stage 1: Detection
        start_time = time.time()
        detection_results = self.detect_corrosion(enhanced)
        detection_time = time.time() - start_time
        
        # Process each detection
        results_data = []
        annotated_image = original_image.copy()
        
        if len(detection_results.boxes) > 0:
            for idx, box in enumerate(detection_results.boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det_conf = box.conf.item()
                
                # Crop ROI
                roi = original_image[y1:y2, x1:x2]
                
                # Stage 2: Classification
                start_cls = time.time()
                grade, cls_conf = self.classify_severity(roi)
                classify_time = time.time() - start_cls
                
                # Calculate area percentage (simplified)
                total_pixels = original_image.shape[0] * original_image.shape[1]
                roi_pixels = (x2 - x1) * (y2 - y1)
                area_percentage = (roi_pixels / total_pixels) * 100
                
                # Store results
                result_info = {
                    'detection_id': idx + 1,
                    'bbox': (x1, y1, x2, y2),
                    'detection_conf': det_conf,
                    'grade': grade,
                    'grade_name': self.grades[grade],
                    'classification_conf': cls_conf,
                    'area_percentage': area_percentage,
                    'detection_time_ms': detection_time * 1000,
                    'classification_time_ms': classify_time * 1000
                }
                results_data.append(result_info)
                
                # Save crop if requested
                if save_crops:
                    crop_filename = f"{output_dir}/crops/roi_{idx+1}_grade{grade}.jpg"
                    cv2.imwrite(crop_filename, roi)
                
                # Annotate image
                color = self.grade_colors[grade]
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
                
                # Add label
                label = f"{self.grades[grade]} ({cls_conf:.2f})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(
                    annotated_image,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
        
        # Save annotated image
        output_filename = f"{output_dir}/annotated_{Path(image_path).name}"
        cv2.imwrite(output_filename, annotated_image)
        
        # Compile final results
        final_results = {
            'image_path': str(image_path),
            'output_path': output_filename,
            'total_detections': len(results_data),
            'detections': results_data,
            'total_inference_time_ms': (detection_time * 1000) + 
                                       sum(r['classification_time_ms'] for r in results_data)
        }
        
        return final_results
    
    def print_results(self, results):
        """Print formatted results"""
        print("\n" + "="*60)
        print(f"📊 CORROSION ANALYSIS RESULTS")
        print("="*60)
        print(f"Image: {results['image_path']}")
        print(f"Total Detections: {results['total_detections']}")
        print(f"Total Processing Time: {results['total_inference_time_ms']:.2f} ms")
        
        if results['total_detections'] > 0:
            print("\nDetailed Results:")
            for det in results['detections']:
                print(f"\n  Detection #{det['detection_id']}:")
                print(f"    Severity: {det['grade_name']}")
                print(f"    Confidence: {det['classification_conf']:.3f}")
                print(f"    Area: {det['area_percentage']:.2f}%")
                print(f"    BBox: {det['bbox']}")
        else:
            print("\n✅ No corrosion detected!")
        
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize system
    monitor = CorrosionMonitor(
        detection_model_path="models/detection_best.pt",
        classification_model_path="models/classification_best.pt"
    )
    
    # Process single image
    test_image = "test_images/sample1.jpg"
    results = monitor.process_image(test_image, output_dir="output")
    monitor.print_results(results)