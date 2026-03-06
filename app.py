from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import librosa
import os
import traceback
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load Audio Model
model_path = "ASV_deepfake_audio_model.h5"
audio_model = load_model(model_path)

# AUDIO FEATURE EXTRACTION
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    
    # Extract 46 MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=46)
    
    # Take mean across time axis
    features = np.mean(mfccs, axis=1, keepdims=True)  # Shape (46,1)
    
    # Reshape for model input
    features = np.expand_dims(features, axis=0)  # Shape (1,46,1)
    
    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = file.filename.lower()

    upload_folder = "static/uploads/"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)

    try:
        if filename.endswith((".wav", ".mp3", ".flac")):
            features = extract_audio_features(file_path)
            prediction = audio_model.predict(features)[0][0]
            
            result = "Fake" if prediction > 0.5 else "Real"

            # Always express confidence in the predicted verdict (0–1 range):
            # - If Fake: confidence = raw sigmoid output  (high = very fake)
            # - If Real: confidence = 1 - raw sigmoid     (high = very real)
            confidence = float(prediction) if result == "Fake" else float(1.0 - prediction)

            return jsonify({
                "type": "audio",
                "prediction": result,
                "confidence": confidence,
                "audio_path": file_path
            })

        else:
            return jsonify({"error": "Unsupported audio format"}), 400

    except Exception as e:
        error_message = traceback.format_exc()
        print("Error Occurred:\n", error_message)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)