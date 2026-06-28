import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from pathlib import Path

app = Flask(__name__, static_folder='../frontend/dist')
CORS(app)

# Models Configuration & Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DETECTOR_PATH = BASE_DIR / "models" / "corrosion_detector.pt"
GRADER_PATH   = BASE_DIR / "models" / "corrosion_grader.pt"

detector = None
classifier = None

def get_models():
    global detector, classifier
    if detector is None or classifier is None:
        print("\n[ML Engine] Lazy loading YOLO models into memory...")
        detector = YOLO(str(DETECTOR_PATH), task='detect')
        classifier = YOLO(str(GRADER_PATH), task='classify')
        print("[ML Engine] Models loaded successfully!")
    return detector, classifier

def preprocess_image(image):
    """ Apply CLAHE contrast enhancement """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Grade definitions & OpenCV warm earthy BGR colors
GRADES = {0: "Healthy", 1: "Mild", 2: "Moderate", 3: "Severe"}
COLORS = {
    0: (50, 125, 46),     # Green - Grade 0 (Healthy) - hex #2E7D32
    1: (3, 129, 183),     # Yellow/Ochre - Grade 1 - hex #B78103
    2: (21, 67, 216),     # Terracotta Orange - Grade 2 - hex #D84315
    3: (40, 40, 198)      # Crimson Red - Grade 3 - hex #C62828
}

# Persistent state
state = {
    "current": {
        "grade": "Grade 0",
        "confidence": 100.0,
        "timestamp": "System Boot",
        "engine": "Initializing..."
    },
    "logs": []
}

@app.route('/api/update', methods=['POST'])
def update_status():
    global state
    data = request.json
    
    # Update the live status
    state["current"] = {
        "grade": data.get("grade", "Grade 0"),
        "confidence": round(data.get("confidence", 0), 2),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "engine": data.get("engine", state["current"]["engine"])
    }
    
    # Keep a running log of the last 10 detections
    state["logs"].insert(0, state["current"])
    if len(state["logs"]) > 10:
        state["logs"].pop()
        
    return jsonify({"message": "Status updated successfully!"}), 200

@app.route('/api/live', methods=['GET'])
def get_live_status():
    return jsonify(state), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    try:
        det_model, cls_model = get_models()
        
        # Determine input type (multipart file or base64 JSON)
        img_bytes = None
        if 'image' in request.files:
            file = request.files['image']
            img_bytes = file.read()
        elif request.json and 'image_b64' in request.json:
            b64_str = request.json['image_b64']
            if ',' in b64_str:
                b64_str = b64_str.split(',')[1]
            img_bytes = base64.b64decode(b64_str)
        else:
            return jsonify({"error": "No image provided"}), 400
        
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        original_img = img.copy()
        annotated_img = img.copy()
        
        # Apply CLAHE preprocessing
        enhanced_img = preprocess_image(original_img)
        
        # Stage 1: Corrosion Detection (YOLOv12)
        det_results = det_model.predict(enhanced_img, conf=0.45, iou=0.45, imgsz=640, verbose=False)[0]
        
        spots = []
        max_grade = 0
        
        if len(det_results.boxes) > 0:
            for box in det_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Minimum ROI size filter (skip tiny noise regions)
                roi_w, roi_h = x2 - x1, y2 - y1
                if roi_w < 30 or roi_h < 30:
                    continue
                
                roi = original_img[y1:y2, x1:x2]
                if roi.size > 0:
                    # Stage 2: Severity Grading (YOLOv8-cls)
                    cls_results = cls_model.predict(roi, imgsz=224, verbose=False)[0]
                    confidence = float(cls_results.probs.top1conf.item())
                    grade = int(cls_results.probs.top1)
                    
                    # Filter out Grade-0 (Healthy) false positives
                    if grade == 0:
                        continue
                    
                    if grade > max_grade:
                        max_grade = grade
                    
                    spots.append({
                        "grade": f"Grade {grade}",
                        "severity": GRADES[grade],
                        "confidence": round(confidence * 100, 2),
                        "box": [x1, y1, x2, y2]
                    })
                    
                    # Draw bold bounding box
                    color = COLORS[grade]
                    label = f"Grade-{grade}: {GRADES[grade]} ({confidence*100:.1f}%)"
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 8)
                    
                    # Bold bounding box text badge
                    font_scale = 0.75
                    font_thickness = 3
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(annotated_img, (x1, y1 - th - 14), (x1 + tw + 10, y1), color, -1)
                    cv2.putText(annotated_img, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                    
        # Encode annotated image to JPEG base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "max_grade": f"Grade {max_grade}",
            "severity": GRADES[max_grade],
            "total_spots": len(spots),
            "spots": spots,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}"
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print("\nCORROSION SENSE AI - Telemetry & Assessment Server")
    print("--------------------------------------------------")
    print("Local access: http://localhost:5000")
    print("Network access: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)