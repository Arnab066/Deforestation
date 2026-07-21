import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .fuzzy_logic import evaluate_fuzzy_risk

def preprocess_sentinel(s2_input, s1_input=None, size=(224, 224)):
    # Read S2 (Optical)
    try:
        s2_img = Image.open(s2_input).convert('RGB')
    except Exception as e:
        raise ValueError(f"Sentinel-2 image invalid or unreadable: {e}")
    
    s2_resized = np.array(s2_img.resize(size, Image.Resampling.LANCZOS))
    
    # Read S1 (SAR)
    if s1_input is not None:
        try:
            s1_img = Image.open(s1_input).convert('L')
        except Exception:
            s1_img = s2_img.convert('L')
    else:
        s1_img = s2_img.convert('L')
            
    s1_resized = np.array(s1_img.resize(size, Image.Resampling.LANCZOS))
    
    return s2_resized, s1_resized

def process_deforestation_fuzzy(s2_rgb, s1_gray, save_path=None):
    # Compute pseudo-NDVI from standard RGB: (NIR=Green channel proxy if 3-band RGB)
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
    
    # Overlay using numpy arrays
    overlay = (s2_rgb.astype(np.float32) * 0.5 + risk_map.astype(np.float32) * 0.5).astype(np.uint8)
    
    # Render overlay to base64 Data URL (serverless friendly)
    buffer = io.BytesIO()
    plt.imsave(buffer, overlay, format='png')
    buffer.seek(0)
    base64_str = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')

    if save_path:
        try:
            plt.imsave(save_path, overlay)
        except Exception:
            pass # fallback on read-only environments
            
    # Compute average risk score
    mean_score = np.mean(risk_scores)
    
    if mean_score >= 60:
        overall_risk = "High"
    elif mean_score >= 35:
        overall_risk = "Medium"
    else:
        overall_risk = "Low"
        
    return overall_risk, base64_str
