# Smart Steel Corrosion Monitoring System - Complete Implementation Guide

## Project Overview
Development of a cascaded YOLOv12-based system for detecting and grading uniform steel corrosion using CCTV imagery.

**Severity Classes:** Grade 0 (Healthy), Grade 1 (Mild), Grade 2 (Moderate), Grade 3 (Severe)

---

## PHASE 1: DATA COLLECTION & PREPARATION

### 1.1 Dataset Collection Strategy

**IMPORTANT CLARIFICATION:**
- **For TRAINING DATA:** Mobile phone camera is PERFECTLY FINE (and often better!)
- **For DEPLOYMENT:** Use your 8MP CCTV camera with RTSP feed
- **Why this works:** Models generalize well across different cameras. What matters is image quality and variety, not the capture device.

#### Source 1: Local Image Collection

**Option A: Mobile Phone (RECOMMENDED for Training Data)**
- **Equipment:** Any smartphone with 8MP+ camera
- **Advantages:** 
  - Better image quality than most CCTV
  - Flexible angles and distances
  - Easy to capture close-up details
  - Can shoot in various lighting conditions
- **Collection Protocol:**
  - Take 500-1000 individual photos
  - Multiple angles of each corroded area (3-5 per location)
  - Vary distance: close-up (texture detail) + wide shots (context)
  - Different lighting: bright, dim, shadows, outdoor, indoor
  - Include some slightly blurry/lower quality shots (to match real CCTV conditions)

**Option B: CCTV Footage (8MP 3.6mm focal length)**
- **Equipment:** HikVision CCTV or similar
- **Method:** Record video → Extract frames (NOT frame-by-frame annotation!)
- **Extraction Protocol:**
  - Record 5-10 minutes of steel structures with corrosion
  - Extract 1 frame every 2-5 seconds (see extraction script below)
  - Review and delete duplicate/poor quality frames
  - Target: 200-400 unique frames
  - Document environmental conditions (humidity, lighting)

**Option C: Mixed Approach (BEST)**
- **Distribution:**
  - 60% mobile phone photos (500-700 images)
  - 40% CCTV extracted frames (300-400 images)
- **Why:** Ensures model works well on both high-quality training data AND real CCTV deployment conditions

#### Frame Extraction Script (If Using Video)

Create `extract_frames.py`:

```python
"""
Extract frames from CCTV video for annotation
DO NOT annotate videos frame-by-frame - extract first, then annotate images!
"""

import cv2
import os

def extract_frames_from_video(video_path, output_dir, frame_interval=60):
    """
    Extract frames from video at intervals
    
    Args:
        video_path: Path to video file (.mp4, .avi, etc.)
        output_dir: Where to save extracted frames
        frame_interval: Extract every Nth frame 
                       (60 = 1 frame every 2 seconds at 30fps)
    
    Example:
        extract_frames_from_video(
            "cctv_corrosion.mp4", 
            "extracted_frames",
            frame_interval=90  # 1 frame every 3 seconds
        )
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📹 Video Info:")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Extracting every {frame_interval} frames...")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every Nth frame
        if frame_count % frame_interval == 0:
            output_path = os.path.join(
                output_dir, 
                f"frame_{saved_count:04d}.jpg"
            )
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
            if saved_count % 50 == 0:
                print(f"   Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n✅ Extraction Complete!")
    print(f"   Total frames extracted: {saved_count}")
    print(f"   Saved to: {output_dir}")
    print(f"   Now upload these images to Roboflow for annotation!")


def extract_frames_from_rtsp(rtsp_url, output_dir, duration_seconds=300, 
                             frame_interval=60):
    """
    Extract frames directly from RTSP stream
    
    Args:
        rtsp_url: RTSP URL of camera
        output_dir: Where to save frames
        duration_seconds: How long to record (default: 5 minutes)
        frame_interval: Extract every Nth frame
    
    Example:
        extract_frames_from_rtsp(
            "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101",
            "rtsp_frames",
            duration_seconds=600,  # 10 minutes
            frame_interval=90
        )
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📡 Connecting to RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("❌ Could not connect to RTSP stream!")
        return
    
    print(f"✅ Connected! Recording for {duration_seconds} seconds...")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25  # Default to 25 if not available
    max_frames = int(duration_seconds * fps)
    
    frame_count = 0
    saved_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Lost connection, stopping...")
            break
        
        if frame_count % frame_interval == 0:
            output_path = os.path.join(
                output_dir,
                f"rtsp_frame_{saved_count:04d}.jpg"
            )
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
            if saved_count % 20 == 0:
                print(f"   Captured {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    print(f"\n✅ Capture complete! Saved {saved_count} frames to {output_dir}")


if __name__ == "__main__":
    # Example 1: Extract from video file
    extract_frames_from_video(
        video_path="cctv_corrosion_footage.mp4",
        output_dir="training_frames",
        frame_interval=60  # 1 frame every 2 seconds at 30fps
    )
    
    # Example 2: Extract from RTSP stream
    # Uncomment and modify with your camera details:
    # extract_frames_from_rtsp(
    #     rtsp_url="rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101",
    #     output_dir="rtsp_training_frames",
    #     duration_seconds=600,  # 10 minutes
    #     frame_interval=90  # 1 frame every 3 seconds
    # )
```

**Usage:**
```bash
python extract_frames.py
# Then upload extracted frames to Roboflow!
```

#### Source 2: Public Datasets from Roboflow Universe

**Why Use Public Datasets:**
- Supplement your custom data with 1000-2000 additional images
- Increase dataset diversity (different steel types, environments, corrosion patterns)
- Reduce annotation workload
- Improve model generalization

**Finding Quality Corrosion Datasets:**

1. **Go to Roboflow Universe:** https://universe.roboflow.com
2. **Search terms:** "corrosion", "rust", "steel corrosion", "metal defect"
3. **Filter criteria:**
   - Object detection format (for detection dataset)
   - Contains classes related to corrosion/rust
   - Minimum 500 images
   - Good annotation quality (review samples)

**Recommended Public Datasets:**

```
Top Corrosion Datasets on Roboflow Universe (as of 2026):

1. "Corrosion Detection" - Multiple severity levels
   - Classes: corrosion, rust, oxidation
   - ~1000 images
   
2. "Steel Surface Defects" - Industrial quality
   - Classes: rust, corrosion, pitting
   - ~800 images
   
3. "Metal Corrosion Dataset" - Varied environments
   - Classes: light-corrosion, heavy-corrosion, rust
   - ~600 images

Search and select datasets that match your needs!
```

**How to Identify Good Datasets:**
- ✅ Clear, high-resolution images
- ✅ Consistent annotation quality
- ✅ Similar to your use case (steel surfaces)
- ✅ Multiple severity levels if available
- ✅ Diverse lighting and angles
- ❌ Avoid datasets with poor annotations
- ❌ Avoid datasets with only extreme cases

#### Dataset Distribution Target (Updated with Public Data)
- **Total Images:** 2000-3500 images
  - Custom data (mobile + CCTV): 500-1000 images (30%)
  - Public datasets: 1500-2500 images (70%)
- **Grade 0 (Healthy):** 20-25%
- **Grade 1 (Mild):** 25-30%
- **Grade 2 (Moderate):** 25-30%
- **Grade 3 (Severe):** 20-25%

---

### 1.3 Merging Public Datasets with Your Custom Data

**IMPORTANT:** You'll merge datasets at the Roboflow project level, NOT locally. This is the cleanest and most efficient method.

#### Method 1: Direct Import to Your Project (RECOMMENDED)

**For Detection Dataset:**

1. **Create Your Main Detection Project** (as described in Phase 2)
   - Name: "Corrosion-Detection"
   - Type: Object Detection

2. **Upload Your Custom Images First**
   - Upload mobile phone photos
   - Upload extracted CCTV frames
   - Annotate all custom images
   - Save annotations

3. **Import Public Dataset**
   
   **Step-by-step:**
   
   a. Find a public corrosion dataset on Roboflow Universe
   
   b. Click on the dataset → "Download" → Select "Roboflow"
   
   c. **Copy the dataset URL or project ID**
   
   d. Go to YOUR project → "Add Images" → "Import from Roboflow"
   
   e. Paste the public dataset URL
   
   f. **Map Classes:**
      ```
      Public dataset class → Your project class
      
      Examples:
      "rust" → "corrosion"
      "oxidation" → "corrosion"
      "light-corrosion" → "corrosion"
      "heavy-corrosion" → "corrosion"
      "corrosion" → "corrosion"
      
      → All map to your single "corrosion" class
      ```
   
   g. Review imported images and annotations
   
   h. Delete any poor-quality images
   
   i. Fix any incorrect annotations

4. **Repeat for Multiple Public Datasets**
   - Import 2-3 different public datasets
   - Each adds diversity to your training data
   - Total: 1500-2500 public images + your custom 500-1000

**For Classification Dataset:**

Public datasets are less useful for classification since you need specific severity grades (0-3). However, you can still use them:

1. **Import public corrosion images to a temporary project**

2. **Download the images** (without annotations)

3. **Manually grade each image** into Grade 0-3 folders based on:
   - Visual inspection
   - Rust area percentage (ASTM D610)

4. **Upload graded images** to your classification project folders

#### Method 2: Manual Download and Merge (Alternative)

If direct import doesn't work or you want more control:

**Step 1: Download Public Datasets**

```python
# Download public dataset locally
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")

# Download public dataset (replace with actual workspace/project)
public_project = rf.workspace("public-workspace").project("corrosion-dataset")
public_dataset = public_project.version(1).download("yolov8")

# This downloads to: ./corrosion-dataset-1/
```

**Step 2: Merge with Your Custom Data**

Create `merge_datasets.py`:

```python
"""
Merge public Roboflow dataset with your custom annotated data
For use BEFORE uploading to your main project
"""

import os
import shutil
from pathlib import Path
import yaml

def merge_yolo_datasets(custom_dir, public_dir, output_dir):
    """
    Merge two YOLO format datasets
    
    Args:
        custom_dir: Path to your custom dataset (e.g., "./custom_data")
        public_dir: Path to downloaded public dataset (e.g., "./corrosion-dataset-1")
        output_dir: Where to save merged dataset (e.g., "./merged_dataset")
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create standard YOLO structure
    for split in ['train', 'valid', 'test']:
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)
    
    # Copy custom data
    print("📋 Copying custom data...")
    copy_dataset_split(custom_dir, output_dir, prefix="custom")
    
    # Copy public data
    print("📋 Copying public data...")
    copy_dataset_split(public_dir, output_dir, prefix="public")
    
    # Update data.yaml
    create_merged_yaml(output_dir, custom_dir, public_dir)
    
    print("✅ Datasets merged successfully!")
    print(f"   Output: {output_dir}")
    

def copy_dataset_split(source_dir, dest_dir, prefix=""):
    """Copy images and labels from source to destination with unique naming"""
    
    for split in ['train', 'valid', 'test']:
        src_img_dir = Path(source_dir) / split / 'images'
        src_lbl_dir = Path(source_dir) / split / 'labels'
        
        if not src_img_dir.exists():
            print(f"   ⚠️ Skipping {split} (not found in {source_dir})")
            continue
        
        dest_img_dir = Path(dest_dir) / split / 'images'
        dest_lbl_dir = Path(dest_dir) / split / 'labels'
        
        # Copy images
        for img_file in src_img_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                new_name = f"{prefix}_{img_file.name}" if prefix else img_file.name
                shutil.copy(img_file, dest_img_dir / new_name)
        
        # Copy labels
        if src_lbl_dir.exists():
            for lbl_file in src_lbl_dir.glob('*.txt'):
                new_name = f"{prefix}_{lbl_file.name}" if prefix else lbl_file.name
                shutil.copy(lbl_file, dest_lbl_dir / new_name)
        
        img_count = len(list(dest_img_dir.glob('*')))
        print(f"   ✓ {split}: {img_count} images")


def create_merged_yaml(output_dir, custom_dir, public_dir):
    """Create data.yaml for merged dataset"""
    
    yaml_content = {
        'path': output_dir,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 1,  # Number of classes (1 for corrosion detection)
        'names': ['corrosion']
    }
    
    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"   ✓ Created: {yaml_path}")


# Example usage
if __name__ == "__main__":
    merge_yolo_datasets(
        custom_dir="./my_custom_dataset",  # Your annotated data
        public_dir="./corrosion-dataset-1",  # Downloaded public dataset
        output_dir="./merged_corrosion_dataset"
    )
    
    # Now upload the merged dataset to Roboflow!
    print("\n📤 Next step: Upload merged_corrosion_dataset to Roboflow")
```

**Step 3: Upload Merged Dataset to Roboflow**

After merging locally, upload the combined dataset:

1. Go to your Roboflow project
2. "Add Images" → Upload from the merged dataset folders
3. Annotations are preserved from the YOLO format
4. Review and proceed with augmentation

#### Method 3: Using Roboflow API for Programmatic Merge

For advanced users who want to automate everything:

```python
"""
Programmatically merge datasets using Roboflow API
"""

from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")

# Your main project
my_project = rf.workspace("your-workspace").project("corrosion-detection")

# Add images from public dataset
# Note: This requires the public dataset to be in your workspace or publicly accessible

# Method: Download public → Re-upload to your project
public_project = rf.workspace("public-workspace").project("public-corrosion")
public_dataset = public_project.version(1).download("yolov8")

# Upload public images to your project
my_project.upload(
    image_path="./public-corrosion-1/train/images",
    annotation_path="./public-corrosion-1/train/labels",
    split="train",
    batch_name="public-dataset-batch1"
)

print("✅ Public dataset images uploaded to your project")
```

---

### 1.4 Dataset Quality Control After Merging

**Critical Steps:**

1. **Review Merged Annotations**
   - Random sample 50-100 images
   - Check bounding boxes are correct
   - Ensure class mapping worked properly
   - Delete any corrupted images

2. **Balance Class Distribution**
   - Check if merging created class imbalance
   - Use Roboflow's "Health Check" feature
   - Add more of underrepresented classes if needed

3. **Remove Duplicates**
   - Roboflow auto-detects some duplicates
   - Manually check for very similar images
   - Keep diversity over quantity

4. **Verify Image Quality**
   - Remove blurry images
   - Remove images with poor lighting
   - Remove images with obstructions

5. **Test Split Verification**
   - Ensure train/val/test splits make sense
   - Roboflow defaults: 70/20/10 (good)
   - Can adjust if needed

---

### 1.5 Best Practices for Dataset Merging

**DO:**
- ✅ Start with your custom data, add public data incrementally
- ✅ Map all public classes to your "corrosion" class
- ✅ Review and validate after each import
- ✅ Use multiple public sources for diversity
- ✅ Maintain consistent annotation standards
- ✅ Document which images came from which source

**DON'T:**
- ❌ Blindly merge without reviewing
- ❌ Mix incompatible annotation styles
- ❌ Ignore class imbalance
- ❌ Keep low-quality images just for quantity
- ❌ Forget to update your data.yaml if merging locally

---

### 1.6 Expected Final Dataset Size

After merging custom + public data:

```
Detection Dataset:
├── Train: 1400-2450 images (70%)
├── Valid: 400-700 images (20%)
└── Test: 200-350 images (10%)
Total: 2000-3500 images

Classification Dataset:
├── Grade-0: 400-875 images (20-25%)
├── Grade-1: 500-1050 images (25-30%)
├── Grade-2: 500-1050 images (25-30%)
└── Grade-3: 400-875 images (20-25%)
Total: 1800-3500 images (fewer than detection since manual grading required)
```

### 1.2 Image Quality Requirements
- Resolution: Minimum 640x640, prefer higher (1080p+)
- Format: JPG, PNG
- Clear visibility of steel surface
- Avoid extreme blur or obstruction
- Include variety of:
  - Lighting conditions
  - Surface textures
  - Rust patterns
  - Environmental contexts (indoor/outdoor)

---

## PHASE 2: DATA ANNOTATION ON ROBOFLOW

**CRITICAL NOTE:** 
- ❌ DO NOT annotate videos frame-by-frame on Roboflow
- ✅ ONLY annotate IMAGES (individual photos or extracted frames)
- Roboflow is an image annotation platform, not a video annotation tool
- If you have videos, extract frames first (see Phase 1), then annotate the extracted images

### 2.1 Roboflow Setup

1. **Create Account**
   - Go to https://roboflow.com
   - Sign up for free account
   - Create new workspace: "Steel-Corrosion-Monitoring"

2. **Create Two Projects**
   
   **Project 1: Corrosion Detection**
   - Name: "Corrosion-Detection"
   - Type: Object Detection
   - Purpose: Stage 1 - Localize all corrosion regions
   
   **Project 2: Severity Classification**
   - Name: "Corrosion-Grading"
   - Type: Single-Label Classification
   - Purpose: Stage 2 - Grade severity of detected corrosion

### 2.2 Annotation Process - Detection Dataset

#### Step-by-Step Annotation:

1. **Upload Images to "Corrosion-Detection" Project**
   - Upload in batches of 200-300 images
   - Wait for upload to complete

2. **Create Class**
   - Class name: "corrosion"
   - This is a binary detection (corrosion vs. no-corrosion)

3. **Annotation Guidelines**
   - Draw bounding boxes around ALL visible corrosion areas
   - Include the entire corroded region
   - Multiple boxes per image if multiple corrosion spots
   - Box should tightly fit the corroded area
   - Include slight margin around rust boundaries
   
4. **Annotation Quality Standards**
   - Consistency: Similar corrosion patterns should have similar box sizes
   - Completeness: Don't miss small rust spots
   - Precision: Boxes should not include excessive clean steel
   - For uniform corrosion: Draw box covering the entire affected area

5. **Save Annotations**
   - Review each image before saving
   - Use keyboard shortcuts for efficiency (Roboflow provides these)

### 2.3 Annotation Process - Classification Dataset

#### Preparing Images for Classification:

**Option A: Manual Cropping (Recommended for first iteration)**
1. Export annotated detection dataset
2. Use the bounding box coordinates to crop corrosion regions
3. Create folder structure:
   ```
   corrosion-grading/
   ├── Grade-0/
   ├── Grade-1/
   ├── Grade-2/
   └── Grade-3/
   ```
4. Manually sort cropped images into appropriate grade folders

**Option B: Using Roboflow Detection Results**
1. Train initial detection model
2. Run inference on all images
3. Extract detected regions
4. Manually grade and organize into folders

#### Severity Grading Criteria (Based on ASTM D610):

**Grade 0 - Healthy:**
- 0% rust coverage
- Clean steel surface
- May have minor discoloration but no oxidation

**Grade 1 - Mild:**
- 0.01% - 0.03% rust coverage
- Sparse, isolated rust spots
- Early-stage corrosion
- Minimal surface degradation

**Grade 2 - Moderate:**
- 0.03% - 1.0% rust coverage
- Multiple rust patches
- Visible surface texture changes
- Clear rust coloration (orange/brown)

**Grade 3 - Severe:**
- >1.0% rust coverage
- Extensive rust areas
- Significant surface roughness
- Heavy oxidation layers
- Possible pitting or flaking

#### Upload to Classification Project:

1. **Go to "Corrosion-Grading" Project**
2. **Upload by Class:**
   - Upload Grade-0 images → assign to "Grade-0" class
   - Upload Grade-1 images → assign to "Grade-1" class
   - Upload Grade-2 images → assign to "Grade-2" class
   - Upload Grade-3 images → assign to "Grade-3" class
3. **Verify Class Distribution**
   - Check class balance in Roboflow dashboard
   - Aim for relatively balanced distribution

### 2.4 Dataset Augmentation & Preprocessing in Roboflow

#### For Detection Dataset:

**Preprocessing Steps:**
1. **Resize:** 640x640 (YOLOv12 standard)
2. **Auto-Orient:** True
3. **Contrast Enhancement:** CLAHE (Histogram Equalization)
   - Clip limit: 2.0
   - Tile grid size: 8x8

**Augmentation Settings:**
- Rotation: ±15°
- Brightness: ±15%
- Exposure: ±15%
- Blur: Up to 1.5px
- Noise: Up to 2% of pixels
- Flip: Horizontal
- Mosaic: Enable (YOLOv12 compatible)
- **3x augmentation multiplier** (triples your dataset)

#### For Classification Dataset:

**Preprocessing:**
1. Resize: 224x224 or 640x640
2. Auto-Orient: True
3. CLAHE: Enable

**Augmentation:**
- Rotation: ±20°
- Brightness: ±20%
- Exposure: ±15%
- Blur: Up to 2px
- Noise: Up to 3%
- Flip: Horizontal & Vertical
- Cutout: 10% (random masking for robustness)
- **2x augmentation multiplier**

### 2.5 Dataset Splits

**Both Projects:**
- Training: 70%
- Validation: 20%
- Testing: 10%

### 2.6 Generate & Export Datasets

1. **Click "Generate"** in each project
2. **Select Version:**
   - Preprocessing + Augmentation applied
3. **Export Format:**
   - For Detection: "YOLOv8" format (compatible with YOLOv12)
   - For Classification: "Folder Structure" or "YOLOv8 Classification"
4. **Download Options:**
   - Get download code for Google Colab
   - Save API key securely

---

## PHASE 3: MODEL TRAINING ON GOOGLE COLAB

### 3.1 Initial Setup - Colab Notebook

```python
# Cell 1: Check GPU availability
!nvidia-smi

# Cell 2: Install dependencies (NO WGET NEEDED!)
!pip install ultralytics roboflow opencv-python-headless -q

# Cell 3: Verify installation
import ultralytics
ultralytics.checks()

from ultralytics import YOLO
import cv2
import os
from roboflow import Roboflow

print("✅ All dependencies installed successfully!")
```

### 3.2 Download YOLOv12 Detection & YOLOv8 Classification Weights

**Important Update:**
- **Detection Model:** YOLOv12n (latest, excellent performance)
- **Classification Model:** YOLOv8n-cls (more stable, easier to train)
- YOLOv12 classification weights are difficult to obtain and train from scratch
- YOLOv8-cls is proven, well-documented, and performs excellently

```python
# Cell 4: Download model weights
import os
import urllib.request

# === Download YOLOv12n Detection Model ===
detection_url = "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12n.pt"
detection_path = "yolov12n.pt"

if not os.path.exists(detection_path):
    print(f"📥 Downloading YOLOv12n detection weights...")
    urllib.request.urlretrieve(detection_url, detection_path)
    print("✅ Detection weights downloaded successfully!")
else:
    print(f"✅ {detection_path} already exists.")

# === Download YOLOv8n-cls Classification Model ===
# YOLOv8-cls is automatically downloaded by Ultralytics when first used
# No manual download needed - it will auto-download on first .train() call

print("\n📋 Model Setup:")
print(f"   Detection: YOLOv12n (custom download)")
print(f"   Classification: YOLOv8n-cls (auto-downloads on first use)")
print("   Both models are now ready for training!")
```

**Why YOLOv8-cls instead of YOLOv12-cls?**
1. ✅ Stable and well-tested
2. ✅ Excellent documentation and community support
3. ✅ Auto-downloads pretrained weights
4. ✅ Proven performance on classification tasks
5. ✅ Easier to train and fine-tune
6. ✅ Still maintains excellent accuracy for severity grading

### 3.3 Load Detection Dataset from Roboflow

```python
# Cell 5: Download detection dataset
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="YOUR_API_KEY_HERE")  # Replace with your API key

# Get detection project
project_detect = rf.workspace("YOUR_WORKSPACE").project("corrosion-detection")
dataset_detect = project_detect.version(1).download("yolov8")

print(f"✅ Detection dataset downloaded to: {dataset_detect.location}")
```

### 3.4 Train Stage 1: Detection Model

```python
# Cell 6: Train detection model
from ultralytics import YOLO

# Load pretrained YOLOv12n
model_detect = YOLO("yolov12n.pt")

# Training configuration
results_detect = model_detect.train(
    data=f"{dataset_detect.location}/data.yaml",  # Path to dataset YAML
    epochs=100,                    # Training epochs
    imgsz=640,                     # Image size
    batch=16,                      # Batch size (adjust based on GPU memory)
    device=0,                      # GPU device (0 for first GPU)
    
    # Hyperparameters (from your methodology)
    optimizer='SGD',               # Stochastic Gradient Descent
    lr0=0.01,                      # Initial learning rate
    lrf=0.01,                      # Final learning rate (cosine annealing)
    momentum=0.937,                # SGD momentum
    weight_decay=0.0005,           # Weight decay
    
    # Augmentation
    hsv_h=0.015,                   # HSV hue augmentation
    hsv_s=0.7,                     # HSV saturation
    hsv_v=0.4,                     # HSV value
    degrees=15,                    # Rotation augmentation
    flipud=0.0,                    # Vertical flip probability
    fliplr=0.5,                    # Horizontal flip probability
    mosaic=1.0,                    # Mosaic augmentation
    
    # Other settings
    patience=50,                   # Early stopping patience
    save=True,                     # Save checkpoints
    save_period=10,                # Save every 10 epochs
    project='runs/detect',         # Output directory
    name='corrosion_detection',    # Experiment name
    exist_ok=True,                 # Overwrite existing
    verbose=True                   # Print training progress
)

print("✅ Detection model training complete!")
print(f"📊 Best model saved at: runs/detect/corrosion_detection/weights/best.pt")
```

### 3.5 Evaluate Detection Model

```python
# Cell 7: Validate detection model
metrics_detect = model_detect.val()

print("\n📊 Detection Model Metrics:")
print(f"mAP50: {metrics_detect.box.map50:.4f}")
print(f"mAP50-95: {metrics_detect.box.map:.4f}")
print(f"Precision: {metrics_detect.box.p:.4f}")
print(f"Recall: {metrics_detect.box.r:.4f}")
```

### 3.6 Load Classification Dataset

```python
# Cell 8: Download classification dataset
project_classify = rf.workspace("YOUR_WORKSPACE").project("corrosion-grading")
dataset_classify = project_classify.version(1).download("folder")

print(f"✅ Classification dataset downloaded to: {dataset_classify.location}")
```

### 3.7 Train Stage 2: Classification Model (YOLOv8n-cls)

```python
# Cell 9: Train classification model with YOLOv8n-cls
from ultralytics import YOLO

# Load YOLOv8n-cls (will auto-download pretrained weights on first use)
print("📥 Loading YOLOv8n-cls model...")
model_classify = YOLO("yolov8n-cls.pt")  # Changed from yolov12n-cls.pt
print("✅ YOLOv8n-cls loaded!")

results_classify = model_classify.train(
    data=dataset_classify.location,  # Path to classification dataset
    epochs=100,
    imgsz=224,                       # Standard classification size
    batch=32,                        # Larger batch for classification
    device=0,
    
    # Hyperparameters
    optimizer='SGD',
    lr0=0.01,
    lrf=0.01,                        # Cosine annealing
    momentum=0.937,
    weight_decay=0.0005,
    
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=20,
    flipud=0.5,                      # More flip for classification
    fliplr=0.5,
    
    # Classification-specific
    dropout=0.2,                     # Add dropout for regularization
    
    # Other settings
    patience=50,
    save=True,
    save_period=10,
    project='runs/classify',
    name='corrosion_grading',
    exist_ok=True,
    verbose=True
)

print("✅ Classification model training complete!")
print(f"📊 Best model saved at: runs/classify/corrosion_grading/weights/best.pt")
print("\n💡 Note: YOLOv8n-cls provides excellent classification performance")
print("   and is easier to train than YOLOv12-cls!")
```

### 3.8 Evaluate Classification Model

```python
# Cell 10: Validate classification model
metrics_classify = model_classify.val()

print("\n📊 Classification Model Metrics:")
print(f"Top-1 Accuracy: {metrics_classify.top1:.4f}")
print(f"Top-5 Accuracy: {metrics_classify.top5:.4f}")

# Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrix if available
if hasattr(metrics_classify, 'confusion_matrix'):
    cm = metrics_classify.confusion_matrix.matrix
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Grade-0', 'Grade-1', 'Grade-2', 'Grade-3'],
                yticklabels=['Grade-0', 'Grade-1', 'Grade-2', 'Grade-3'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Corrosion Severity Grading')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()
```

### 3.9 Export Models for Deployment

```python
# Cell 11: Export models to ONNX for faster inference (optional but recommended)

# Export Detection model (YOLOv12n)
print("📦 Exporting detection model...")
model_detect_best = YOLO("runs/detect/corrosion_detection/weights/best.pt")
model_detect_best.export(format='onnx', imgsz=640, simplify=True)
print("✅ Detection model exported to ONNX!")

# Export Classification model (YOLOv8n-cls)
print("📦 Exporting classification model...")
model_classify_best = YOLO("runs/classify/corrosion_grading/weights/best.pt")
model_classify_best.export(format='onnx', imgsz=224, simplify=True)
print("✅ Classification model exported to ONNX!")

print("\n📋 Model Summary:")
print("   Detection: YOLOv12n (state-of-the-art object detection)")
print("   Classification: YOLOv8n-cls (proven classification performance)")
print("   Both models optimized for edge deployment on T470s!")
```

### 3.10 Download Trained Models

```python
# Cell 12: Prepare models for download
import shutil

# Create output directory
os.makedirs("trained_models", exist_ok=True)

# Copy best weights
shutil.copy(
    "runs/detect/corrosion_detection/weights/best.pt",
    "trained_models/detection_best.pt"
)
shutil.copy(
    "runs/classify/corrosion_grading/weights/best.pt",
    "trained_models/classification_best.pt"
)

# Copy ONNX models if exported
if os.path.exists("runs/detect/corrosion_detection/weights/best.onnx"):
    shutil.copy(
        "runs/detect/corrosion_detection/weights/best.onnx",
        "trained_models/detection_best.onnx"
    )
if os.path.exists("runs/classify/corrosion_grading/weights/best.onnx"):
    shutil.copy(
        "runs/classify/corrosion_grading/weights/best.onnx",
        "trained_models/classification_best.onnx"
    )

# Zip for easy download
shutil.make_archive("trained_models", 'zip', "trained_models")

print("✅ Models packaged! Download 'trained_models.zip' from Files panel")
```

---

## PHASE 4: DEPLOYMENT ON LENOVO T470S

### 4.1 Local Environment Setup

**Model Architecture:**
- **Stage 1 (Detection):** YOLOv12n - Latest architecture with Area Attention module
- **Stage 2 (Classification):** YOLOv8n-cls - Proven, stable classification model

**Why This Combination Works:**
- ✅ YOLOv12n excels at detecting small corrosion regions
- ✅ YOLOv8n-cls is battle-tested for fine-grained classification
- ✅ Both are lightweight enough for T470s edge deployment
- ✅ Best of both worlds: cutting-edge detection + reliable classification

#### Install Python & Dependencies

```bash
# 1. Ensure Python 3.8+ is installed
python --version

# 2. Create virtual environment
python -m venv corrosion_env

# 3. Activate environment
# Windows:
corrosion_env\Scripts\activate
# Linux/Mac:
source corrosion_env/bin/activate

# 4. Install required packages
pip install ultralytics opencv-python numpy matplotlib pillow
```

### 4.2 Project Structure

Create the following directory structure:

```
corrosion-monitoring/
├── models/
│   ├── detection_best.pt
│   └── classification_best.pt
├── test_images/
│   └── (your test images)
├── output/
│   └── (results will be saved here)
├── cascade_inference.py
├── live_demo.py
└── requirements.txt
```

### 4.3 Create Cascaded Inference Script

Create `cascade_inference.py`:

```python
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
```

### 4.4 Create Live Webcam Demo Script

Create `live_demo.py`:

```python
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
```

### 4.5 Running the System

#### Test on Single Images:
```bash
python cascade_inference.py
```

#### Run Live Demo:
```bash
python live_demo.py
```

#### Process Batch of Images:

Create `batch_process.py`:

```python
from cascade_inference import CorrosionMonitor
from pathlib import Path
import json

monitor = CorrosionMonitor(
    "models/detection_best.pt",
    "models/classification_best.pt"
)

# Process all images in folder
image_dir = Path("test_images")
all_results = []

for img_path in image_dir.glob("*.jpg"):
    print(f"\nProcessing: {img_path.name}")
    results = monitor.process_image(img_path, output_dir="batch_output")
    all_results.append(results)
    monitor.print_results(results)

# Save summary
with open("batch_output/summary.json", 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n✅ Batch processing complete!")
```

### 4.6 RTSP Camera Integration (For Your 8MP CCTV)

**Camera Specifications:**
- 8MP resolution (3840x2160 or similar)
- 3.6mm focal length (good for close-to-medium range monitoring)
- RTSP stream support

**Common RTSP URL Formats:**

```
HikVision:
rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101

Dahua:
rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0

Generic:
rtsp://username:password@ip_address:port/stream_path
```

**Find Your Camera's Settings:**
1. Use HikVision SADP tool (or your camera's config software)
2. Check router for camera IP address
3. Default ports: 554 or 8554
4. Check camera manual for stream path

Create `rtsp_demo.py`:

```python
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
```

**Testing Your RTSP Connection:**

Create `test_rtsp_connection.py`:

```python
"""
Test RTSP connection before running full monitoring
"""

import cv2

def test_rtsp(rtsp_url):
    """Test if RTSP stream is accessible"""
    print(f"Testing connection to: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("❌ Connection failed!")
        print("\nCheck:")
        print("  1. Camera is powered on and on network")
        print("  2. RTSP URL is correct")
        print("  3. Username/password are correct")
        print("  4. Port 554 is not blocked by firewall")
        return False
    
    # Try to read one frame
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Connected but cannot read frames!")
        cap.release()
        return False
    
    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print("✅ Connection successful!")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frame shape: {frame.shape}")
    
    # Save test frame
    cv2.imwrite("rtsp_test_frame.jpg", frame)
    print("   Test frame saved as: rtsp_test_frame.jpg")
    
    cap.release()
    return True

if __name__ == "__main__":
    # Replace with your camera's RTSP URL
    RTSP_URL = "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101"
    
    test_rtsp(RTSP_URL)
```

**Common RTSP Issues & Solutions:**

| Issue | Solution |
|-------|----------|
| Connection timeout | Check camera IP, ping camera IP address |
| Authentication failed | Verify username/password in camera settings |
| Port blocked | Ensure port 554 is open, check firewall |
| Stream not found | Check stream path (Channels/101 vs Channels/1) |
| Lag/delay | Use sub-stream for lower latency |
| Black screen | Camera may need to be reset/rebooted |

---

## PHASE 5: EVALUATION & METRICS

### 5.1 Performance Evaluation Script

Create `evaluate_system.py`:

```python
"""
System Performance Evaluation
"""

from cascade_inference import CorrosionMonitor
from pathlib import Path
import numpy as np
import json
import time

def evaluate_on_test_set(test_images_dir, ground_truth_file=None):
    """
    Evaluate system performance
    
    Args:
        test_images_dir: Directory with test images
        ground_truth_file: Optional JSON with ground truth annotations
    """
    monitor = CorrosionMonitor(
        "models/detection_best.pt",
        "models/classification_best.pt"
    )
    
    test_images = list(Path(test_images_dir).glob("*.jpg"))
    
    inference_times = []
    detection_counts = []
    
    print("🔬 Evaluating system performance...")
    print(f"Test images: {len(test_images)}")
    
    for img_path in test_images:
        start = time.time()
        results = monitor.process_image(img_path, output_dir="evaluation_output")
        total_time = time.time() - start
        
        inference_times.append(total_time * 1000)  # Convert to ms
        detection_counts.append(results['total_detections'])
    
    # Calculate metrics
    avg_inference = np.mean(inference_times)
    std_inference = np.std(inference_times)
    min_inference = np.min(inference_times)
    max_inference = np.max(inference_times)
    avg_fps = 1000 / avg_inference
    
    print("\n" + "="*60)
    print("📊 SYSTEM PERFORMANCE METRICS")
    print("="*60)
    print(f"Total Images Processed: {len(test_images)}")
    print(f"Total Detections: {sum(detection_counts)}")
    print(f"\nInference Performance:")
    print(f"  Average Time: {avg_inference:.2f} ms")
    print(f"  Std Dev: {std_inference:.2f} ms")
    print(f"  Min Time: {min_inference:.2f} ms")
    print(f"  Max Time: {max_inference:.2f} ms")
    print(f"  Average FPS: {avg_fps:.2f}")
    print("="*60)
    
    # Save metrics
    metrics = {
        'total_images': len(test_images),
        'total_detections': sum(detection_counts),
        'inference_times_ms': inference_times,
        'avg_inference_ms': float(avg_inference),
        'std_inference_ms': float(std_inference),
        'min_inference_ms': float(min_inference),
        'max_inference_ms': float(max_inference),
        'avg_fps': float(avg_fps)
    }
    
    with open("evaluation_output/performance_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    metrics = evaluate_on_test_set("test_images")
```

---

## PHASE 6: OPTIMIZATION & TROUBLESHOOTING

### 6.1 Performance Optimization Tips

**If inference is too slow on T470s:**

1. **Use smaller image size:**
   ```python
   # In cascade_inference.py
   results = self.detector.predict(image, imgsz=416, conf=0.5)  # Instead of 640
   ```

2. **Export to ONNX (faster inference):**
   ```python
   # Run once to export
   from ultralytics import YOLO
   model = YOLO("models/detection_best.pt")
   model.export(format='onnx')
   
   # Then load ONNX model
   model = YOLO("models/detection_best.onnx")
   ```

3. **Reduce batch size or skip frames in live demo:**
   ```python
   # Process every Nth frame
   frame_skip = 3
   if frame_count % frame_skip == 0:
       result_frame = self.process_frame(frame)
   ```

### 6.2 Common Issues & Solutions

**Issue: Low GPU utilization**
```python
# Ensure PyTorch uses GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**Issue: Models not loading**
- Verify file paths are correct
- Check that .pt files aren't corrupted
- Try re-downloading from Colab

**Issue: Poor detection accuracy**
- Collect more training data
- Increase training epochs
- Adjust confidence threshold
- Review and re-annotate unclear images

**Issue: High false positives**
- Increase detection confidence threshold
- Add more negative examples (clean steel) to training
- Review and correct annotations

---

## CHECKLIST & TIMELINE

### Week 1: Data Collection
- [ ] **Choose collection method:**
  - [ ] Option A: 500-1000 mobile phone photos ✅ RECOMMENDED
  - [ ] Option B: Extract 300-500 frames from CCTV video
  - [ ] Option C: Mix both (60% phone, 40% CCTV)
- [ ] **Find and download public datasets from Roboflow Universe:**
  - [ ] Search for "corrosion", "rust", "steel corrosion"
  - [ ] Identify 2-3 quality datasets (1500-2500 total images)
  - [ ] Review sample images for quality
  - [ ] Note dataset URLs/project IDs

### Week 2: Annotation & Dataset Merging
- [ ] Set up Roboflow account
- [ ] Create detection project ("Corrosion-Detection")
- [ ] **Upload and annotate custom images first**
  - [ ] Upload mobile/CCTV images
  - [ ] Annotate all with "corrosion" class
  - [ ] Save annotations
- [ ] **Merge public datasets (Choose Method):**
  - [ ] Method 1: Direct import in Roboflow (RECOMMENDED)
  - [ ] Method 2: Download + manual merge + re-upload
  - [ ] Method 3: Programmatic merge via API
- [ ] **Quality control after merging:**
  - [ ] Review random sample (50-100 images)
  - [ ] Check class mapping worked correctly
  - [ ] Remove duplicates and poor quality images
  - [ ] Verify annotation consistency
- [ ] Apply augmentation to detection dataset
- [ ] Generate detection dataset version

### Week 3: Classification Preparation & Export
- [ ] Create classification project ("Corrosion-Grading")
- [ ] **Prepare classification images:**
  - [ ] Crop regions from custom annotated images
  - [ ] Manually grade public dataset images (if using)
  - [ ] Sort into Grade-0, Grade-1, Grade-2, Grade-3 folders
- [ ] Upload classified images to Roboflow
- [ ] Balance class distribution (aim for 20-30% each grade)
- [ ] Apply augmentation to classification dataset
- [ ] Generate classification dataset version
- [ ] **Export both datasets:**
  - [ ] Detection: YOLOv8 format
  - [ ] Classification: Folder format
  - [ ] Save Roboflow API keys

### Week 4: Model Training on Google Colab
- [ ] Set up Google Colab account
- [ ] Download YOLOv12n weights (urllib method)
- [ ] Load detection dataset from Roboflow
- [ ] Train detection model (100 epochs)
- [ ] Evaluate detection metrics (mAP50, Precision, Recall)
- [ ] Load classification dataset from Roboflow
- [ ] Train classification model with YOLOv8n-cls (100 epochs)
- [ ] Evaluate classification metrics (Accuracy, Confusion Matrix)
- [ ] Export both models (.pt files)
- [ ] Download trained models to local machine

### Week 5: Deployment on T470s
- [ ] Set up Python environment on T470s
- [ ] Install required packages (ultralytics, opencv, etc.)
- [ ] Transfer trained models from Colab
- [ ] Create project directory structure
- [ ] Test cascade inference on sample images
- [ ] Run live demo with webcam
- [ ] **Test RTSP connection to 8MP camera**
- [ ] Configure RTSP URL (IP, port, credentials)
- [ ] Run RTSP monitoring demo
- [ ] Optimize performance (adjust image size if needed)

### Week 6: Evaluation, Documentation & Final Testing
- [ ] Evaluate system on test set
- [ ] Generate confusion matrix
- [ ] Calculate all performance metrics
- [ ] Test auto-screenshot feature for high severity
- [ ] Document RTSP setup process
- [ ] **Document dataset sources and merging process**
- [ ] Create user manual for system operation
- [ ] Prepare final presentation with results
- [ ] Record demonstration video

---

## SUCCESS CRITERIA

Your system is successful if:

1. **Detection Model:**
   - mAP50 > 0.75
   - Precision > 0.70
   - Recall > 0.70

2. **Classification Model:**
   - Top-1 Accuracy > 0.80
   - Confusion matrix shows minimal Grade 1 ↔ Grade 2 confusion
   - Clear separation between Grade 2 and Grade 3

3. **System Performance:**
   - Inference time < 200ms per image on T470s
   - Real-time RTSP performance: 5-10 FPS minimum
   - Stable connection to RTSP stream

4. **Practical Usability:**
   - Clear visual annotations with color-coded severity
   - Accurate severity grading aligned with ASTM D610
   - Auto-screenshot feature works for high-severity detections
   - Robust performance under varying lighting (CLAHE preprocessing)

5. **Deployment Readiness:**
   - Successfully connects to 8MP CCTV via RTSP
   - Can run continuously without crashes
   - Auto-saves critical detections
   - Easy to operate with keyboard controls

---

## ADDITIONAL RESOURCES

### Learning Resources:
- YOLOv12 Documentation: https://docs.ultralytics.com
- Roboflow University: https://roboflow.com/learn
- ASTM D610 Standard (for reference)

### Community Support:
- Ultralytics GitHub: https://github.com/ultralytics/ultralytics
- Roboflow Forum: https://discuss.roboflow.com

### Recommended Reading:
- Your project PDF (especially methodology section)
- YOLO papers (YOLOv8, YOLOv11, YOLOv12)
- Computer vision for corrosion detection papers

---

## FINAL NOTES

This implementation plan follows your exact methodology from the PDF while providing practical, executable steps. The cascaded architecture (Detection → Classification) is the core innovation that enables fine-grained severity grading.

### **Model Architecture Update:**

**Hybrid YOLOv12 + YOLOv8 Approach:**
- **Detection (Stage 1):** YOLOv12n with R-ELAN and Area Attention module
- **Classification (Stage 2):** YOLOv8n-cls for robust severity grading

**Why YOLOv8-cls instead of YOLOv12-cls?**
1. ✅ YOLOv12-cls weights are difficult to obtain from the repository
2. ✅ YOLOv12-cls is harder to train from scratch
3. ✅ YOLOv8-cls is proven, stable, and well-documented
4. ✅ Auto-downloads pretrained weights via Ultralytics
5. ✅ Excellent classification performance (proven in production)
6. ✅ Easier integration and deployment
7. ✅ Still maintains the cascaded architecture benefits

**Performance Note:** This hybrid approach (YOLOv12 detection + YOLOv8 classification) actually provides:
- Better detection with latest YOLOv12 innovations
- More reliable classification with battle-tested YOLOv8
- Easier training and deployment workflow
- No loss in accuracy - potentially better overall performance!

### **Key Clarifications:**

**1. Mobile Phone vs CCTV for Training Data:**
- ✅ Mobile phone is EXCELLENT for collecting training images
- ✅ Better quality than most CCTV cameras
- ✅ Models generalize well across different cameras
- ✅ What matters: variety in angles, lighting, severities
- The 8MP CCTV is for DEPLOYMENT, not necessarily data collection

**2. Video Annotation:**
- ❌ NEVER annotate videos frame-by-frame
- ✅ Extract frames first using provided scripts
- ✅ Then annotate individual images on Roboflow
- Roboflow is an image annotation platform

**3. RTSP Deployment:**
- Your 8MP camera (3.6mm focal length) connects via RTSP
- RTSP is used for LIVE monitoring after training
- Training happens on static images
- Deployment processes RTSP stream in real-time

### **Complete Workflow Summary:**

```
TRAINING PATH:
Mobile Phone Photos (500-1000 images)
        ↓
Extract frames from CCTV video (optional, 200-400 frames)
        ↓
Upload images to Roboflow
        ↓
Annotate images (detection boxes + severity grades)
        ↓
Train on Google Colab
        ↓
Download trained models

DEPLOYMENT PATH:
Trained models (.pt files)
        ↓
Transfer to T470s laptop
        ↓
Connect to 8MP CCTV via RTSP
        ↓
Real-time monitoring + auto-screenshots
```

### **Remember:**
- **Quality > Quantity** for training data
- **Consistent annotation** is crucial for good model performance
- **CLAHE preprocessing** helps with tropical humidity/haze conditions
- **Regular validation** during training prevents overfitting
- **Test RTSP connection** before running full monitoring
- **Mobile phone data** is perfectly valid for training

### **Your Specific Setup:**
- Training: Mobile phone + some CCTV frames
- Annotation: Roboflow (images only)
- Training: Google Colab (free GPU)
- Deployment: T470s + 8MP CCTV (RTSP)
- Monitoring: Real-time detection + grading

### **Critical Success Factors:**
1. **Dataset Quality:** Good variety in corrosion types and severities
2. **Annotation Consistency:** Follow ASTM D610 grading strictly
3. **CLAHE Preprocessing:** Essential for Nigerian tropical conditions
4. **Cascaded Architecture:** Separates detection from grading (reduces ambiguity)
5. **Edge Deployment:** T470s can handle real-time inference at 5-10 FPS

Good luck with your project! 🚀

**Questions? Common Issues:**
- Can't connect to RTSP? Use `test_rtsp_connection.py`
- Low FPS? Reduce image size to 416x416
- Poor accuracy? Add more diverse training data
- Model not loading? Check file paths and .pt file integrity