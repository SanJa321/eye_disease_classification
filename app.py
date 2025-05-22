from flask import Flask, render_template, request, jsonify
import os
import requests
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model configuration
MODEL_PATH = "modelLast2_save.keras"
MODEL_URL = "https://drive.google.com/file/d/16zAOYXQUYFvBZazWrEITPcTDFcdKC0TC/view?usp=drive_link"  # <-- Replace with actual public URL

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from URL...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Model downloaded.")

# Load Keras model
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ['CNV', 'DME', 'Drusen', 'Normal', "not an oct image"]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Load and preprocess image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    predicted_class = class_labels[np.argmax(preds[0])]

    return jsonify({
        'prediction': predicted_class,
        'file_path': file_path
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
