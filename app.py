import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from utils.preprocess import preprocess_sentinel, process_deforestation_fuzzy

app = Flask(__name__)
os.makedirs('static', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image_s2' not in request.files:
        return jsonify({'error': 'No Sentinel-2 image uploaded'}), 400
        
    file_s2 = request.files['image_s2']
    if file_s2.filename == '':
        return jsonify({'error': 'No selected S2 file'}), 400
        
    upload_path_s2 = os.path.join('uploads', 's2_' + file_s2.filename)
    file_s2.save(upload_path_s2)
    
    # Handle Sentinel-1 file if provided, otherwise fallback to using S2 for simulation
    upload_path_s1 = upload_path_s2 
    if 'image_s1' in request.files:
        file_s1 = request.files['image_s1']
        if file_s1.filename != '':
            upload_path_s1 = os.path.join('uploads', 's1_' + file_s1.filename)
            file_s1.save(upload_path_s1)
    
    try:
        # Preprocess both files
        s2_rgb, s1_gray = preprocess_sentinel(upload_path_s2, upload_path_s1)
        
        output_filename = 'fuzzy_output_' + file_s2.filename
        output_path = os.path.join('static', output_filename)
        
        # Run Fuzzy Inference System
        risk = process_deforestation_fuzzy(s2_rgb, s1_gray, output_path)
            
        return jsonify({
            'risk': risk,
            'output_image': output_filename
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5005)
