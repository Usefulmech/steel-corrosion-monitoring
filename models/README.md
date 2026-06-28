# Model Weights

Model weights are not stored in Git due to size limitations.

## Download Instructions

After training on Kaggle, download and rename:
- `corrosion_detector.pt` - YOLOv12n detection model (rename from `detection_best.pt`)
- `corrosion_grader.pt` - YOLOv8n-cls classification model (rename from `classification_best.pt`)

Place them in this directory.

## File Sizes
- corrosion_detector.pt: ~6 MB
- corrosion_grader.pt: ~3 MB

## Training Details
- See notebooks/ folder for training code
- See docs/results.md for performance metrics