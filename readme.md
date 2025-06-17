# Infant Cry Audio Classification App

A cross-platform (mobile/web) application for classifying infant cry audio using machine learning. This project leverages a React Native/Expo frontend and a Flask backend, powered by a TFLite audio classification model. The app enables users to upload or record infant cry audio, analyzes the audio, and predicts the type of cry (e.g., hungry, tired, discomfort, etc.).

---

## Features

- **Mobile & Web Support:** Built with React Native (Expo) for seamless deployment on both web and mobile devices.
- **Audio Upload/Recording:** Users can upload or record audio files for analysis.
- **ML-Powered Prediction:** Audio is processed and classified using a TensorFlow Lite (TFLite) model served by a Flask backend.
- **Consistent ML Pipeline:** Feature extraction and model input/output are consistent across training, backend, and app.
- **LSTM RNN Model:** Includes code for training, evaluating, and exporting an LSTM-based audio classifier.

---

## Folder Structure

```
project-root/
│
├── app/                # React Native/Expo frontend
│   ├── app/            # App screens and navigation
│   ├── assets/         # Model files, fonts, images
│   └── ...
│
├── backend/            # Flask backend API
│   └── app.py          # Main backend server
│
├── model/              # ML model training and assets
│   ├── working_analysis.ipynb  # Main notebook for data prep, training, export
│   ├── audio_prediction_model.tflite  # Dense NN TFLite model
│   ├── lstm_audio_model.tflite        # LSTM TFLite model
│   ├── donateacry_corpus/     # Audio dataset
│   └── ...
│
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## How to Run

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd <project-root>
```

### 2. Set Up the Python Backend

- Create and activate a Python virtual environment:
  ```sh
  python3 -m venv myenv
  source myenv/bin/activate
  ```
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```
- Start the Flask server:
  ```sh
  cd backend
  python app.py
  ```
  The backend will run on `http://0.0.0.0:5050` by default.

### 3. Set Up the Frontend (Expo App)

- Install dependencies:
  ```sh
  cd app
  npm install
  ```
- Start the Expo app:
  ```sh
  npx expo start
  ```
- Follow the Expo CLI instructions to run on web, Android, or iOS.

### 4. Connect Frontend and Backend

- The frontend is configured to send requests to the backend's `/predict` endpoint.
- For mobile testing, use a public tunnel (e.g., [ngrok](https://ngrok.com/)) to expose your backend:
  ```sh
  ngrok http 5050
  ```
- Update the backend URL in the frontend code if needed.

---

## Model Training & Export

- All model training, evaluation, and TFLite export code is in `model/working_analysis.ipynb`.
- The notebook covers:
  - Data preparation and feature extraction (MFCC, spectral centroid, ZCR, chroma)
  - Label encoding
  - Training Random Forest, XGBoost, Dense NN, and LSTM models
  - Exporting models to TFLite for deployment
  - Testing TFLite inference for consistency

---

## Key Technologies

- **Frontend:** React Native, Expo
- **Backend:** Flask, TensorFlow Lite, Librosa
- **ML:** TensorFlow/Keras, scikit-learn, XGBoost

---

## Credits

- Dataset: [Donate a Cry Corpus](https://zenodo.org/record/1203745)
- Inspired by research in infant cry analysis and audio ML.

---

## License

This project is for educational and research purposes. See the LICENSE file for details.
