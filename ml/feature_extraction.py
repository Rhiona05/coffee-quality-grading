import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Paths
PREPROCESSED_PATH = "ml/preprocessed"   # preprocessed images folder
OUTPUT_CSV = "ml/features.csv"          # where features will be saved

# Parameters
GLCM_DISTANCES = [1]       # pixel distances
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # angles
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'

# Collect feature rows
feature_list = []

# Loop over each grade folder
for grade_folder in os.listdir(PREPROCESSED_PATH):
    grade_path = os.path.join(PREPROCESSED_PATH, grade_folder)
    if not os.path.isdir(grade_path):
        continue
    
    for img_file in os.listdir(grade_path):
        img_path = os.path.join(grade_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping corrupt image: {img_file}")
            continue
        
        # GLCM
        glcm = graycomatrix((img*255).astype(np.uint8), distances=GLCM_DISTANCES, angles=GLCM_ANGLES, levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        energy = graycoprops(glcm, 'energy').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # LBP
        lbp = local_binary_pattern(img, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
        lbp_mean = lbp.mean()
        lbp_std = lbp.std()
        
        # Append row
        feature_list.append({
            'contrast': contrast,
            'energy': energy,
            'homogeneity': homogeneity,
            'correlation': correlation,
            'lbp_mean': lbp_mean,
            'lbp_std': lbp_std,
            'label': grade_folder
        })

# Convert to DataFrame
df = pd.DataFrame(feature_list)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Feature extraction complete! Saved to {OUTPUT_CSV}")