import React, { useState } from "react";
import {
  Button,
  View,
  Text,
  ActivityIndicator,
  Alert,
  Platform,
} from "react-native";
import * as DocumentPicker from "expo-document-picker";

export default function PredictScreen() {
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Add state for file upload
  const [selectedFile, setSelectedFile] = useState<any>(null);

  // File input handler for web
  function handleFileChangeWeb(event: any) {
    const file = event.target.files[0];
    if (file) setSelectedFile(file);
  }

  // File picker for native
  async function handlePickNative() {
    setResult(null);
    setSelectedFile(null);
    const res = await DocumentPicker.getDocumentAsync({
      type: "audio/*", // Restrict to audio files
      copyToCacheDirectory: true,
      multiple: false,
    });
    if (!res.canceled && res.assets && res.assets.length > 0) {
      setSelectedFile(res.assets[0]);
      Alert.alert("File Selected", res.assets[0].name || "Audio file selected");
    }
  }

  // Upload and predict handler
  async function handleUploadAndPredict() {
    if (!selectedFile) {
      Alert.alert("No file selected", "Please select a .wav file first.");
      return;
    }
    setResult(null);
    setLoading(true);
    try {
      const formData = new FormData();
      if (Platform.OS === "web") {
        formData.append("file", selectedFile);
      } else {
        // Use fetch with uri for React Native
        // @ts-ignore
        formData.append("file", {
          uri: selectedFile.uri,
          name: selectedFile.name || "audio.wav",
          type: selectedFile.type || "audio/wav",
        } as any);
      }
      const response = await fetch(
        "https://8057-2404-7c00-41-6b55-c510-d9e6-e92b-48d0.ngrok-free.app/predict",
        {
          method: "POST",
          body: formData,
        }
      );
      const data = await response.json();
      setLoading(false);
      if (data.error) {
        Alert.alert("Prediction Error", data.error);
      } else {
        setResult(`Predicted Label: ${data.predicted_label}`);
      }
    } catch (e: any) {
      setLoading(false);
      Alert.alert("Error", e.message || String(e));
    }
  }

  return (
    <View
      style={{
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
        backgroundColor: "#fff",
      }}
    >
      {Platform.OS === "web" ? (
        <input
          type="file"
          accept="audio/wav,audio/*"
          onChange={handleFileChangeWeb}
          style={{ marginBottom: 16 }}
        />
      ) : (
        <Button title="Pick Audio File" onPress={handlePickNative} />
      )}
      {selectedFile && (
        <Text style={{ marginBottom: 8, color: "#0a7ea4" }}>
          File: {selectedFile.name || selectedFile.uri}
        </Text>
      )}
      <Button title="Analyse" onPress={handleUploadAndPredict} />
      {loading && (
        <ActivityIndicator
          size="large"
          color="#0a7ea4"
          style={{ marginTop: 20 }}
        />
      )}
      {result && <Text style={{ marginTop: 20, fontSize: 18 }}>{result}</Text>}
    </View>
  );
}
