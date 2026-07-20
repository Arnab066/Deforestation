import os
import tempfile
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from utils.preprocess import preprocess_sentinel, process_deforestation_fuzzy

app = Flask(__name__)

TEMP_DIR = tempfile.gettempdir()

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
        
    upload_path_s2 = os.path.join(TEMP_DIR, 's2_' + file_s2.filename)
    file_s2.save(upload_path_s2)
    
    upload_path_s1 = upload_path_s2 
    if 'image_s1' in request.files:
        file_s1 = request.files['image_s1']
        if file_s1.filename != '':
            upload_path_s1 = os.path.join(TEMP_DIR, 's1_' + file_s1.filename)
            file_s1.save(upload_path_s1)
    
    output_path = None
    try:
        # Preprocess both files
        s2_rgb, s1_gray = preprocess_sentinel(upload_path_s2, upload_path_s1)
        
        output_filename = 'fuzzy_output_' + file_s2.filename + '.png'
        output_path = os.path.join(TEMP_DIR, output_filename)
        
        # Run Fuzzy Inference System
        risk = process_deforestation_fuzzy(s2_rgb, s1_gray, output_path)
        
        # Convert generated output image to Base64 for serverless safety
        with open(output_path, 'rb') as f:
            encoded_img = base64.b64encode(f.read()).decode('utf-8')
        base64_image = f'data:image/png;base64,{encoded_img}'
            
        return jsonify({
            'risk': risk,
            'output_image': base64_image
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary files from /tmp
        for path in [upload_path_s2, upload_path_s1, output_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

if __name__ == '__main__':
    app.run(debug=True, port=5005)
