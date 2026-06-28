"""
Standalone Single Image Tester for Cascaded YOLO Corrosion Pipeline.
Perfect for generating high-quality annotated images for thesis documentation.
"""

import cv2
from ultralytics import YOLO
from pathlib import Path

# --- Configuration & Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent

# Pointing to your highly optimized OpenVINO models
DETECTOR_PATH = BASE_DIR / "models" / "corrosion_detector_openvino_model"
GRADER_PATH   = BASE_DIR / "models" / "corrosion_grader_openvino_model"

# Change this to the path of the image you want to test!
TEST_IMAGE_PATH = BASE_DIR / "results" / "sample_input" /"IMG_20260418_090843_702.jpg"
OUTPUT_DIR = BASE_DIR / "results" / "sample_output"

# Ensure output directory exists for your thesis screenshots
OUTPUT_DIR.mkdir(exist_ok=True)

# Grade definitions (Pure 0-3 scale)
GRADES = {0: "Healthy", 1: "Mild", 2: "Moderate", 3: "Severe"}
COLORS = {
    0: (0, 255, 0),      # Green
    1: (0, 255, 255),    # Yellow
    2: (0, 165, 255),    # Orange
    3: (0, 0, 255)       # Red
}

def preprocess_image(image):
    """ Apply CLAHE contrast enhancement """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def main():
    print(f"Loading OpenVINO Models...")
    detector = YOLO(str(DETECTOR_PATH), task='detect')
    classifier = YOLO(str(GRADER_PATH), task='classify')

    print(f"Loading Image: {TEST_IMAGE_PATH}")
    if not TEST_IMAGE_PATH.exists():
        print("Error: Test image not found! Please check the path.")
        return

    # Read and copy the image
    original_img = cv2.imread(str(TEST_IMAGE_PATH))
    annotated_img = original_img.copy()

    # Preprocess and Detect (Stage 1)
    enhanced_img = preprocess_image(original_img)
    print(" Scanning for corrosion (Stage 1)...")
    
    # Optimized thresholds: conf=0.45 kills weak false positives, iou=0.45 merges duplicate boxes
    det_results = detector.predict(enhanced_img, conf=0.45, iou=0.45, imgsz=640, verbose=False)[0]

    if len(det_results.boxes) == 0:
        print("Status: CLEAR. No corrosion detected.")
        cv2.putText(annotated_img, "STATUS: CLEAR", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        print(f" Found {len(det_results.boxes)} rusted regions. Grading now...")
        
        # Crop and Classify (Stage 2)
        for box in det_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Min ROI size filter: skip patches smaller than 30x30px (noise)
            roi_w, roi_h = x2 - x1, y2 - y1
            if roi_w < 30 or roi_h < 30:
                print(f"   -> Skipping micro-detection ({roi_w}x{roi_h}px)")
                continue
            
            roi = original_img[y1:y2, x1:x2]
            
            if roi.size > 0:
                cls_results = classifier.predict(roi, imgsz=224, verbose=False)[0]
                confidence = cls_results.probs.top1conf.item() * 100
                grade = cls_results.probs.top1
                
                # Filter out Grade-0 (Healthy) — if classifier says healthy, it's a false positive
                if grade == 0:
                    print(f"   -> Region classified as Healthy ({confidence:.1f}%), skipping annotation")
                    continue

                # Draw thick, high-visibility bounding boxes
                color = COLORS[grade]
                label = f"Grade-{grade}: {GRADES[grade]} ({confidence:.1f}%)"
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 10)
                
                # Draw filled background behind text for maximum readability
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.rectangle(annotated_img, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
                cv2.putText(annotated_img, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                
                print(f"   -> Region diagnosed as: {label}")

    # Save the output for your documentation
    output_path = OUTPUT_DIR / f"analyzed_{TEST_IMAGE_PATH.name}"
    cv2.imwrite(str(output_path), annotated_img)
    print(f"\n Annotated image saved successfully to: {output_path}")

    # Display the result on screen
    cv2.imshow("Static Image Analysis", annotated_img)
    print("Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()