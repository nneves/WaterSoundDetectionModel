import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load TFLite model
# interpreter = tf.lite.Interpreter(model_path="model_improved_int8_3_2.tflite")
interpreter = tf.lite.Interpreter(model_path="model_improved_int8_3_2_1.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Configuration (same as training)
chunk_duration = 3  # seconds
sample_rate = 16000
samples_per_chunk = chunk_duration * sample_rate

def predict_wave_file_tflite(audio_path):
    """Predict liters consumption for a wave file using TFLite"""
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Split into chunks
    num_chunks = len(y) // samples_per_chunk
    if num_chunks == 0:
        print(f"Audio too short: {audio_path}")
        return None
    
    predictions_per_chunk = []
    print("=== Detailed per-chunk MFCC/quantization debug ===")
    
    for i in range(num_chunks):
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        chunk = y[start:end]
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13).T
        # Debug stats: before normalization
        pre_min = float(np.min(mfcc))
        pre_max = float(np.max(mfcc))
        pre_mean = float(np.mean(mfcc))
        pre_range = pre_max - pre_min
        print(f"Chunk {i+1} MFCC (before norm) - Min: {pre_min:.4f}, Max: {pre_max:.4f}, Mean: {pre_mean:.4f}, Range: {pre_range:.4f}")
        
        # Normalize
        mfcc = mfcc.astype("float32")
        mfcc = mfcc / (np.max(np.abs(mfcc)) + 1e-6)
        # Debug stats: after normalization
        post_min = float(np.min(mfcc))
        post_max = float(np.max(mfcc))
        post_mean = float(np.mean(mfcc))
        post_range = post_max - post_min
        print(f"Chunk {i+1} MFCC (after norm) - Min: {post_min:.4f}, Max: {post_max:.4f}, Mean: {post_mean:.4f}, Range: {post_range:.4f}")
        
        # Pad sequence
        mfcc_padded = pad_sequences([mfcc], maxlen=200, padding='post', dtype='float32')
        
        # Quantize input for INT8 model
        input_scale, input_zero_point = input_details[0]['quantization']
        mfcc_quantized = (mfcc_padded / input_scale + input_zero_point).astype(np.int8)
        # Debug stats: quantized values
        q_min = int(np.min(mfcc_quantized))
        q_max = int(np.max(mfcc_quantized))
        print(f"Chunk {i+1} Quantized - Min: {q_min}, Max: {q_max}, Scale: {input_scale:.6f}, Zero: {int(input_zero_point)}")
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], mfcc_quantized)
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction and dequantize
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_scale, output_zero_point = output_details[0]['quantization']
        prediction = (output_data.astype(np.float32) - output_zero_point) * output_scale
        print(f"Chunk {i+1} - Prediction: {float(prediction[0][0]):.4f} liters")
        predictions_per_chunk.append(float(prediction[0][0]))
    
    total_liters = sum(predictions_per_chunk)
    
    return {
        'file': audio_path,
        'chunks': num_chunks,
        'total_liters': total_liters,
        'predictions_per_chunk': predictions_per_chunk
    }

# Usage
if __name__ == "__main__":
    audio_file = "audio_samples_3_2_1/0.71l_1753549434_7.wav"
    result = predict_wave_file_tflite(audio_file)
    
    if result:
        print(f"File: {result['file']}")
        print(f"Total predicted: {result['total_liters']:.2f} liters")
        print(f"Chunks processed: {result['chunks']}")
        print("Predictions per chunk:")
        for i, pred in enumerate(result['predictions_per_chunk']):
            print(f"  Chunk {i+1}: {pred:.4f} liters")

# --- Check TFLite schema version ---
import struct

def get_tflite_schema_version(tflite_path):
    with open(tflite_path, "rb") as f:
        buf = f.read()
        version = struct.unpack_from('<i', buf, 4)[0]
        print(f"TFLite schema version: {version}")

get_tflite_schema_version("model_improved_int8_3_2_1.tflite")

# --- Print first 16 bytes of the TFLite file for debugging ---
with open("model_improved_int8_3_2_1.tflite", "rb") as f:
    first16 = f.read(16)
    print("First 16 bytes:", first16)