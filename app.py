import os
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU

# Load Model (Ensure model file is present)
MODEL_PATH = "model/model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "captured_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return "TensorFlow Model API Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({'error': 'No image data received'}), 400

    image_data = data["image"].split(",")[-1]  # Handle base64 prefix
    image_path = os.path.join(UPLOAD_FOLDER, "captured_image.png")
    with open(image_path, "wb") as image_file:
        image_file.write(base64.b64decode(image_data))

    # Load and preprocess image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    result = '🔥 Fire 🔥' if prediction < 0.5 else 'Not Fire'
    return jsonify({'result': result})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
