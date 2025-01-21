from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import cv2
import os
import base64
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return jsonify({'file_path': file_path})
    return jsonify({'error': 'File upload failed'})

@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    img1_path = data['img1_path']
    img2_base64 = data['img2_base64']

    # Load uploaded image
    img1 = cv2.imread(img1_path)

    # Decode base64 image
    img2_data = base64.b64decode(img2_base64)
    np_img2 = np.frombuffer(img2_data, np.uint8)
    img2 = cv2.imdecode(np_img2, cv2.IMREAD_COLOR)

    # Perform face comparison
    result = DeepFace.verify(img1, img2)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)