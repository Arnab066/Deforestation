import cv2
import numpy as np
import matplotlib.pyplot as plt
from .fuzzy_logic import evaluate_fuzzy_risk

def preprocess_sentinel(s2_path, s1_path, size=(224, 224)):
    # Read S2 (Optical)
    s2_img = cv2.imread(s2_path)
    if s2_img is None:
        raise ValueError("Sentinel-2 image not found")
    s2_rgb = cv2.cvtColor(s2_img, cv2.COLOR_BGR2RGB)
    s2_resized = cv2.resize(s2_rgb, size)
    
    # Read S1 (SAR)
    s1_img = cv2.imread(s1_path, cv2.IMREAD_GRAYSCALE)
    if s1_img is None:
        # If no separate S1 file, simulate SAR backscatter from S2 grayscale intensity for demo purposes
        s1_img = cv2.cvtColor(s2_img, cv2.COLOR_BGR2GRAY)
    s1_resized = cv2.resize(s1_img, size)
    
    return s2_resized, s1_resized

def process_deforestation_fuzzy(s2_rgb, s1_gray, save_path):
    # Compute pseudo-NDVI from standard generic true color RGB: 
    # (assuming Green=NIR for mock data if true NIR band missing)
    # R=0 (Red), G=1 (used as proxy for NIR if real S2 data not uploaded)
    r = s2_rgb[:,:,0].astype(np.float32)
    nir = s2_rgb[:,:,1].astype(np.float32) 
    
    ndvi = (nir - r) / (nir + r + 1e-5)
    
    # Normalize SAR backscatter logic (0 to 1)
    s1_norm = s1_gray.astype(np.float32) / 255.0
    
    # Run Fuzzy logic
    risk_scores = evaluate_fuzzy_risk(ndvi, s1_norm)
    
    risk_map = np.zeros((*s2_rgb.shape[:2], 3), dtype=np.uint8)
    
    # High risk >= 60 (Red)
    risk_map[risk_scores >= 60] = [255, 0, 0]
    # Medium risk 35-60 (Yellow)
    risk_map[(risk_scores >= 35) & (risk_scores < 60)] = [255, 255, 0]
    # Low risk < 35 (Green)
    risk_map[risk_scores < 35] = [0, 255, 0]
    
    overlay = cv2.addWeighted(s2_rgb, 0.5, risk_map, 0.5, 0)
    plt.imsave(save_path, overlay)
    
    # Compute average risk score of the central area
    mean_score = np.mean(risk_scores)
    
    if mean_score >= 60:
        overall_risk = "High"
    elif mean_score >= 35:
        overall_risk = "Medium"
    else:
        overall_risk = "Low"
        
    return overall_risk
