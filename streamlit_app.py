import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tempfile

st.set_page_config(page_title="Infant Cry Audio Classifier", layout="centered")
st.title("👶 Infant Cry Audio Classifier (ANN Model)")
st.write("Upload a .wav audio file of an infant's cry to predict the type of cry.")

# Load model and label encoder only once
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("ann_model.keras")
    train_df = pd.read_csv("final_smote.csv")
    le = LabelEncoder()
    le.fit(train_df["label"])
    return model, le

model, le = load_model_and_encoder()

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

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    try:
        features = extract_audio_features(tmp_path)
        input_data = features.reshape(1, -1).astype(np.float32)
        pred_probs = model.predict(input_data)
        pred_label = np.argmax(pred_probs, axis=1)[0]
        pred_label_name = le.inverse_transform([pred_label])[0]
        st.success(f"Predicted Cry Type: **{pred_label_name}**")
        st.write("Class Probabilities:")
        prob_dict = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(pred_probs[0])}
        st.json(prob_dict)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
