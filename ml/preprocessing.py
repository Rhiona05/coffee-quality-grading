import os
import cv2
import numpy as np

# Path to dataset
DATASET_PATH = "dataset"        # your dataset folder
OUTPUT_PATH = "ml/preprocessed" # where preprocessed images will go

# Create output folder if not exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Standard image size
IMG_SIZE = (128, 128)

# Loop over each grade folder
for grade_folder in os.listdir(DATASET_PATH):
    grade_path = os.path.join(DATASET_PATH, grade_folder)
    if not os.path.isdir(grade_path):
        continue
    
    # Create folder in preprocessed path
    output_grade_path = os.path.join(OUTPUT_PATH, grade_folder)
    os.makedirs(output_grade_path, exist_ok=True)
    
    # Process each image
    for img_file in os.listdir(grade_path):
        img_path = os.path.join(grade_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Resize
        img = cv2.resize(img, IMG_SIZE)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Normalize
        gray = gray / 255.0
        
        # Optional: Gaussian Blur to remove noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Save preprocessed image
        save_path = os.path.join(output_grade_path, img_file)
        # Multiply by 255 because cv2.imwrite expects 0-255
        cv2.imwrite(save_path, (gray*255).astype(np.uint8))

print("✅ Preprocessing complete!")