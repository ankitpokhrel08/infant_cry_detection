import streamlit as st
import numpy as np
import librosa
import pickle
import pandas as pd

# Load the trained model and label encoder
with open("audio_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoder using the same classes as training
# We'll infer classes from the CSV if not saved separately
df = pd.read_csv("final.csv")
labels = sorted(df["label"].unique())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(labels)

def extract_audio_features(audio_path, n_mfcc=13):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        features = np.concatenate([
            mfccs, spectral_centroid.ravel(), zcr.ravel(), chroma
        ])
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

st.title("Infant Cry Audio Prediction")
st.write("Upload an infant cry audio file (.wav) to predict the cry category.")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

# --- Updated: Audio recording feature instructions ---
st.write("Or record audio using an external tool (e.g., your phone or https://voicecoach.ai/recorder), then upload the .wav file above.")

# --- New: In-app audio recording using st.audio_input (Streamlit >=1.32) ---
audio_value = st.audio_input("Or record a voice message")

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    features = extract_audio_features("temp.wav")
    if features is not None:
        prediction = model.predict([features])[0]
        predicted_label = le.inverse_transform([prediction])[0]
        st.success(f"Predicted Label: **{predicted_label}**")
    else:
        st.error("Could not extract features from the audio file.")
elif audio_value:
    st.audio(audio_value)
    if st.button("Predict from Recording"):
        # Save recorded audio to a temporary file
        with open("recorded_temp.wav", "wb") as f:
            f.write(audio_value.read())
        features = extract_audio_features("recorded_temp.wav")
        if features is not None:
            prediction = model.predict([features])[0]
            predicted_label = le.inverse_transform([prediction])[0]
            st.success(f"Predicted Label: **{predicted_label}**")
        else:
            st.error("Could not extract features from the recorded audio file.")