import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

# Load TFLite model
TFLITE_MODEL_PATH = "model_float16.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_audio(audio_path):
    """Convert audio file to mel-spectrogram and return processed array."""
    try:
        y, sr = librosa.load(audio_path, sr=22050)  # Load audio
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB scale

        # Resize to match model input
        input_shape = input_details[0]['shape']  # Example: (1, 128, 128, 1)
        mel_spec_db = np.resize(mel_spec_db, (input_shape[1], input_shape[2]))
        mel_spec_db = np.expand_dims(mel_spec_db, axis=(0, -1))  # Add batch & channel dim

        return mel_spec_db.astype(np.float32)

    except Exception as e:
        print("Error processing audio:", str(e))
        return None

def predict_audio(audio_path):
    """Run inference on an audio file and print prediction."""
    input_data = preprocess_audio(audio_path)
    if input_data is None:
        print("Invalid audio format. Please provide a valid audio file.")
        return

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Determine deepfake probability
    is_deepfake = prediction > 0.5
    print(f"Prediction: {'Deepfake' if is_deepfake else 'Real'}")
    print(f"Confidence Score: {prediction:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if an audio file is deepfake or real.")
    parser.add_argument("audio_file", type=str, help=r"C:\Users\navee\Downloads\orgaudio.opus")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
