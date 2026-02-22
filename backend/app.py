import os
import numpy as np
import librosa
from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Path to your ANN model
ANN_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/large_dataset/ann_model.keras')
# Path to the CSV used for label encoding
CSV_PATH = os.path.join(os.path.dirname(__file__), '../model/large_dataset/final_smote.csv')

# Load ANN model
ann_model = tf.keras.models.load_model(ANN_MODEL_PATH)

# Load label encoder
train_df = pd.read_csv(CSV_PATH)
le = LabelEncoder()
le.fit(train_df["label"])

# Feature extraction function (must match training)
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
        input_data = features.reshape(1, -1).astype(np.float32)
        pred_probs = ann_model.predict(input_data)
        pred_label = np.argmax(pred_probs, axis=1)[0]
        pred_label_name = le.inverse_transform([pred_label])[0]
        return jsonify({
            'features': features.tolist(),
            'predicted_label_index': int(pred_label),
            'predicted_label': pred_label_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
