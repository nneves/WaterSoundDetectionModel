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
    
    for i in range(num_chunks):
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        chunk = y[start:end]
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13).T
        
        # Normalize
        mfcc = mfcc.astype("float32")
        mfcc = mfcc / (np.max(np.abs(mfcc)) + 1e-6)
        
        # Pad sequence
        mfcc_padded = pad_sequences([mfcc], maxlen=200, padding='post', dtype='float32')
        
        # Quantize input for INT8 model
        input_scale, input_zero_point = input_details[0]['quantization']
        mfcc_quantized = (mfcc_padded / input_scale + input_zero_point).astype(np.int8)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], mfcc_quantized)
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction and dequantize
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_scale, output_zero_point = output_details[0]['quantization']
        prediction = (output_data.astype(np.float32) - output_zero_point) * output_scale
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
    audio_file = "audio_samples_3_2/3.90l_1753737940_6.wav" # 3.53L => got 3.54L
    # audio_file = "audio_samples_3_2/4.44l_1753550540_3.wav" # 3.55L => got 3.55L
    result = predict_wave_file_tflite(audio_file)
    
    if result:
        print(f"File: {result['file']}")
        print(f"Total predicted: {result['total_liters']:.2f} liters")
        print(f"Chunks processed: {result['chunks']}")
        print("Predictions per chunk:")
        for i, pred in enumerate(result['predictions_per_chunk']):
            print(f"  Chunk {i+1}: {pred:.4f} liters")