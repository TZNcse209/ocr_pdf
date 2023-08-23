# Pipeline:
Step 1: Convert pdf file to image
Step 2: Text detection and draw the bounding box
Step 3: Merge overlapping, nearby bounding boxes
Step 4: Text recognition

# Option 1: 
Text detection using the Character Region Awareness for Text Detection (CRAFT) algorithm and Text recognition using EasyOCR tool
# Option 2: 
Text detection using Character Region Awareness for Text Detection (CRAFT) algorithm and Text recognition using Tesseract engine
# Option 3:
Text detection using a Differentiable Binarization (DB) algorithm and Text recognition using PaddleOCR tool

# Reference:
CRAFT algorithm:
Paper: Character Region Awareness for Text Detection
Link: https://arxiv.org/pdf/1904.01941.pdf
DB algorithm:
Paper: Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion:
Link: https://arxiv.org/pdf/2202.10304.pdf
