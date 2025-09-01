import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Escolha o modelo que pretende usar
model = tf.keras.models.load_model("model_improved_3.2.keras")

# Configurações
chunk_duration = 3  # seconds
sample_rate = 16000
samples_per_chunk = chunk_duration * sample_rate

def predict_wave_file(audio_path):
    """Predict liters consumption for a wave file"""
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Split into chunks
    num_chunks = len(y) // samples_per_chunk
    if num_chunks == 0:
        print(f"Audio too short: {audio_path}")
        return None
    
    X_chunks = []
    
    for i in range(num_chunks):
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        chunk = y[start:end]
        
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13).T
        
        # Normalize
        mfcc = mfcc.astype("float32")
        mfcc = mfcc / (np.max(np.abs(mfcc)) + 1e-6)
        
        X_chunks.append(mfcc)
    
    # Pad sequences
    X_padded = pad_sequences(X_chunks, maxlen=200, padding='post', dtype='float32')
    X = np.array(X_padded)
    
    # Predict all chunks at once
    predictions = model.predict(X, verbose=0)
    total_liters = np.sum(predictions)
    
    return {
        'file': audio_path,
        'chunks': num_chunks,
        'total_liters': float(total_liters),
        'predictions_per_chunk': predictions.flatten().tolist()
    }

if __name__ == "__main__":
    # Exemplo de uso
    # audio_file = "audio_samples_3_2/3.90l_1753737940_6.wav"
    audio_file = "audio_samples_3_2/4.44l_1753550540_3.wav"
    result = predict_wave_file(audio_file)
    
    if result:
        print(f"File: {result['file']}")
        print(f"Total predicted: {result['total_liters']:.2f} liters")
        for i, pred in enumerate(result['predictions_per_chunk']):
            print(f"  Chunk {i+1}: {pred:.4f} liters")