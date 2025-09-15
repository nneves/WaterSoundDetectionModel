import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import time

# Load your trained model
model = tf.keras.models.load_model("model_3.0.0.keras")

# Sampling rate (must match training)
sr = 16000  
duration = 3  # seconds

def capture_audio():
    print("🎙️  Recording 3s of audio...")
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def preprocess_audio(y):
    # Extract 20 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T
    mfcc = mfcc.astype("float32")
    mfcc = mfcc / (np.max(np.abs(mfcc)) + 1e-6)

    # Pad or truncate to 200 time steps
    if mfcc.shape[0] < 200:
        pad_width = 200 - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:200, :]

    return np.expand_dims(mfcc, axis=0)  # (1, 200, 20)

# Live loop
total_predicted = 0.0
print("🟢 Starting live prediction loop. Press Ctrl+C to stop.")
try:
    while True:
        y = capture_audio()
        X_input = preprocess_audio(y)
        liters = model.predict(X_input, verbose=0)[0][0] - 0.20
        total_predicted += liters
        print(f"🚿 Predicted: {liters:.2f} L | Total: {total_predicted:.2f} L\n")
        time.sleep(0.5)  # short delay before next 3-sec chunk
except KeyboardInterrupt:
    print("🛑 Stopped by user.")
