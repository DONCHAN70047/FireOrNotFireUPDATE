import os
# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU
import tensorflow as tf
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

# Check if model exists
MODEL_PATH = "tmp/model/model.h5"  # Your original model path

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

UPLOAD_FOLDER = "captured_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html', output=None)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)

    if not data or "image" not in data:
        return jsonify({'error': 'No image data received'}), 400

    image_data = data["image"]

    if "," in image_data:
        image_data = image_data.split(",")[1]

    try:
        image_path = os.path.join(UPLOAD_FOLDER, "captured_image.png")
        with open(image_path, "wb") as image_file:
            image_file.write(base64.b64decode(image_data))

        print(f"Image saved at: {image_path}")

        # Load and preprocess image
        #image_path
        img = image.load_img('https://github.com/DONCHAN70047/FireOrNotFireUPDATE/blob/main/static/ProjectFire/Testing/fire/abc169.jpg', target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        print(f"Image shape: {img_array.shape}")  # Debugging: Check input shape

        # Make prediction
        prediction = model.predict(img_array)
        print(f"Raw Prediction: {prediction}")  # Debugging: Print model output

        # If model output is a probability (0-1), round it
        if isinstance(prediction, (np.ndarray, list)):
            prediction = np.round(prediction[0][0])

        print(f"Final Prediction (rounded): {prediction}")

        if prediction < 0.5:
            result = '🔥🔥🔥🔥🔥🔥🔥 Fire 🔥🔥🔥🔥🔥🔥🔥'
        else:
            result = '......Not Fire......'

        return jsonify({'result': result})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
