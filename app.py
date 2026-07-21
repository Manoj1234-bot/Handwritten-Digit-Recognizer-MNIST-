"""
Day 11 - Handwritten Digit Recognizer (Flask Web App)
Core Skills: Neural networks, Flask, image preprocessing
Author: Manoj S

Serves a web page with an HTML5 canvas. You draw a digit, it gets sent
to this Flask server as a base64 image, preprocessed to match MNIST's
format (28x28 grayscale), and fed into the trained CNN for prediction.

Requires digit_model.keras to exist - run train_model.py first.
"""

import base64
import io
import os

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps

app = Flask(__name__)

MODEL_PATH = "digit_model.keras"
model = None


def get_model():
    """Lazy-load the model so the app can still start even if training
    hasn't been run yet (gives a clear error instead of crashing)."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"'{MODEL_PATH}' not found. Run 'python train_model.py' first."
            )
        # Import here so the app doesn't need TensorFlow just to show
        # a friendly error message if the model file is missing.
        from tensorflow import keras
        model = keras.models.load_model(MODEL_PATH)
    return model


def preprocess_canvas_image(data_url):
    """
    Convert the canvas's base64 PNG data into a 28x28 normalized array
    matching the format the model was trained on (MNIST style: white
    digit on black background, 0-1 normalized, shape (1, 28, 28, 1)).
    """
    header, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale

    # The canvas draws BLACK strokes on a WHITE background by default,
    # but MNIST digits are WHITE strokes on a BLACK background.
    # Invert colors so it matches what the model expects.
    image = ImageOps.invert(image)

    # Resize to 28x28 (MNIST's native resolution)
    image = image.resize((28, 28))

    array = np.array(image).astype("float32") / 255.0
    array = array.reshape(1, 28, 28, 1)

    return array


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "No image data received."}), 400

        processed = preprocess_canvas_image(image_data)

        clf = get_model()
        predictions = clf.predict(processed, verbose=0)[0]

        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit])

        all_probabilities = {str(i): float(predictions[i]) for i in range(10)}

        return jsonify({
            "predicted_digit": predicted_digit,
            "confidence": round(confidence * 100, 2),
            "all_probabilities": all_probabilities,
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)