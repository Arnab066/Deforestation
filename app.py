import os
import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, jsonify, render_template
from utils.preprocess import preprocess_sentinel, process_deforestation_fuzzy

app = Flask(__name__)

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
        
    file_s1 = None
    if 'image_s1' in request.files:
        temp_s1 = request.files['image_s1']
        if temp_s1.filename != '':
            file_s1 = temp_s1
    
    try:
        # Preprocess directly from file streams without saving to read-only disk
        s2_rgb, s1_gray = preprocess_sentinel(file_s2, file_s1)
        
        # Run Fuzzy Inference System
        risk, output_base64 = process_deforestation_fuzzy(s2_rgb, s1_gray)
            
        return jsonify({
            'risk': risk,
            'output_image': output_base64
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5005)
