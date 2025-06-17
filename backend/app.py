import os
import numpy as np
import librosa
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Path to your TFLite model
TFLITE_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../app/assets/build_model/audio_prediction_model.tflite')

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Feature extraction function (same as training)
def extract_audio_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    features = np.concatenate([
        mfccs, spectral_centroid.ravel(), zcr.ravel(), chroma
    ])
    return features.astype(np.float32)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    temp_path = 'temp.wav'
    file.save(temp_path)
    try:
        features = extract_audio_features(temp_path)
        # Reshape for model input
        input_data = np.expand_dims(features, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = int(np.argmax(output_data))
        return jsonify({'features': features.tolist(), 'predicted_label': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
