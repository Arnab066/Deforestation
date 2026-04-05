<div align="center">
  <h1>🌲 AI Deforestation Risk Prediction System</h1>
  <p><strong>A Fuzzy-Logic Remote Sensing Engine using Sentinel-1 (SAR) & Sentinel-2 (Optical) Data</strong></p>

  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
  [![Flask](https://img.shields.io/badge/Flask-Backend-000000.svg)](https://flask.palletsprojects.com/)
  [![skfuzzy](https://img.shields.io/badge/scikit--fuzzy-Inference-success.svg)](#)
  [![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-red.svg)](https://opencv.org/)

</div>

---

## 📖 Overview
The **AI Deforestation Risk Prediction System** analyzes satellite imagery to calculate real-time deforestation risks and visualizes them on an intuitive drag-and-drop web dashboard. 

Originally structured around Deep Learning, the core engine has been upgraded to utilize **Multimodal Remote Sensing Fuzzy Logic**. By actively cross-referencing pseudo-NDVI metrics (Sentinel-2 Optical) with backscatter structure mappings (Sentinel-1 SAR), the system outputs precise, explainable risk boundaries without the black-box opacity of standard CNNs.

---

## ✨ Features
- **Dual Modal Processing**: Support for Sentinel-2 (Multispectral Optical) and Sentinel-1 (Synthetic Aperture Radar) fusion.
- **Fuzzy Inference System (FIS)**: Implements formalized Zadeh max-min logic via `scikit-fuzzy` to determine deforestation metrics logically.
- **Dynamic Risk Map Overlay**: Instantly overlays intuitive boundary layers:
  - 🔴 **High Risk** (Severe clearance/low veg & backscatter)
  - 🟡 **Medium Risk** (Degradation/Mixed)
  - 🟢 **Low Risk** (Maintained dense canopy)
- **Modern User Interface**: Drag-and-drop responsive frontend built purely with Vanilla CSS/JS without page reloads.

---

## 🛠️ Technology Stack
* **Language:** Python
* **Backend Framework:** Flask
* **Math & Inference:** NumPy, Scikit-Fuzzy
* **Image Processing:** OpenCV (`opencv-python`), Matplotlib
* **Web Frontend:** HTML5, CSS3, JavaScript (Fetch API)

---

## 📁 System Architecture

```text
├── app.py                      # Main Flask Web Server & Routing
├── requirements.txt            # Python Dependencies
├── utils/
│   ├── preprocess.py           # Extracts NDVI and handles Normalization operations
│   └── fuzzy_logic.py          # Vectorized Mamdani FIS Engine
├── static/
│   ├── style.css               # Vanilla CSS Design System 
│   └── script.js               # Dual-Dropzone File Uploader Logic
└── templates/
    └── index.html              # Frontend DOM 
```

---

## 🚀 Getting Started

### 1. Requirements Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Arnab066/Deforestation.git
cd Deforestation
pip install -r requirements.txt
```

### 2. Run the Engine
Start up the local web development server:
```bash
python app.py
```
*Navigate to the address below in your web browser!*
```text
http://127.0.0.1:5000
```

### 3. Usage
Simply drag and drop an Optical (Sentinel-2) image to the first slot. Optionally, provide a corresponding SAR image mapping to the second slot. Hit **Evaluate Fuzzy Risk** and observe the live overlay map classification!

---

<div align="center">
  <i>Developed for scalable, local deforestation tracking.</i>
</div>
