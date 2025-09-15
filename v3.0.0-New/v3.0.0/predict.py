import numpy as np
import librosa
import tensorflow as tf
import re
from pathlib import Path

# Load the model
model = tf.keras.models.load_model("model_3.0.0.keras")

# Parameters from original script_predict.py
sr = 16000  # Sampling rate
duration = 3  # seconds
samples_per_chunk = sr * duration

def extract_liters_from_filename(filename):
    """Extract the expected liters value from the filename pattern (e.g., '1.95l_1750712105_1.wav')"""
    match = re.search(r'(\d+\.\d+)l_', filename)
    if match:
        return float(match.group(1))
    return None

def preprocess_audio(y):
    """
    Preprocess a single chunk of audio using parameters from original script
    """
    # Extract 20 MFCCs (from original script)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T
    mfcc = mfcc.astype("float32")
    mfcc = mfcc / (np.max(np.abs(mfcc)) + 1e-6)

    # Pad or truncate to 200 time steps (from original script)
    if mfcc.shape[0] < 200:
        pad_width = 200 - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:200, :]

    return np.expand_dims(mfcc, axis=0)  # (1, 200, 20)

def predict_water_consumption(audio_path):
    """
    Predict water consumption for a given audio file
    """
    # Load audio
    y, _ = librosa.load(audio_path, sr=sr)
    
    # Split into 3-second chunks
    num_chunks = len(y) // samples_per_chunk
    predictions = []
    
    for i in range(num_chunks):
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        chunk = y[start:end]
        
        # Preprocess chunk
        X_input = preprocess_audio(chunk)
        
        # Make prediction
        liters = model.predict(X_input, verbose=0)[0][0]
        predictions.append(liters)
    
    # Sum all chunk predictions
    total_liters = sum(predictions)
    return total_liters, predictions

def analyze_audio_file(audio_file):
    """Analyze a single audio file and return results"""
    try:
        # Get expected liters from filename
        expected_liters = extract_liters_from_filename(audio_file)
        
        # Make prediction
        predicted_total, chunk_predictions = predict_water_consumption(audio_file)
        
        # Calculate error
        error = abs(predicted_total - expected_liters)
        error_percentage = (error / expected_liters) * 100 if expected_liters else None
        
        return {
            'file': audio_file,
            'expected': expected_liters,
            'predicted': predicted_total,
            'error': error,
            'error_percentage': error_percentage,
            'chunks': chunk_predictions
        }
    except Exception as e:
        print(f"❌ Error processing {audio_file}: {str(e)}")
        return None

if __name__ == "__main__":
    # Test files
    test_files = [
        "dataset_sound/audio_samples/1.48l_min_1.wav",
        "dataset_sound/audio_samples/1.95l_1750712105_1.wav",
        "dataset_sound/audio_samples/2.60l_1750712105_0.wav",
        "dataset_sound/audio_samples/3.75l_min_1.wav"
    ]
    
    print("\n🔍 Running predictions on multiple files...")
    print("=" * 60)
    
    total_error_percentage = 0
    valid_results = 0
    
    for audio_file in test_files:
        result = analyze_audio_file(audio_file)
        if result:
            print(f"\n🎵 File: {Path(result['file']).name}")
            print(f"📊 Expected: {result['expected']:.2f}L | Predicted: {result['predicted']:.2f}L")
            print(f"📈 Error: {result['error']:.2f}L ({result['error_percentage']:.1f}%)")
            print("\nChunk predictions:")
            for i, pred in enumerate(result['chunks'], 1):
                print(f"   Chunk {i}: {pred:.2f}L")
            print("-" * 60)
            
            total_error_percentage += result['error_percentage']
            valid_results += 1
    
    if valid_results > 0:
        avg_error = total_error_percentage / valid_results
        print(f"\n📊 Average error across all files: {avg_error:.1f}%")