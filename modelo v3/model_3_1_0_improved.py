import os
import re
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Diretório com os ficheiros de áudio (.wav, .mp3, etc.)
audio_dir = "dataset_sound/audio_samples_3_1"
X_total = []
y_total = []

chunk_duration = 3  # seconds
sample_rate = 16000
samples_per_chunk = chunk_duration * sample_rate

for fname in os.listdir(audio_dir):
    if fname.endswith((".wav", ".mp3")):
        path = os.path.join(audio_dir, fname)

        # Carregar o áudio
        y, sr = librosa.load(path, sr=sample_rate)

        # Extrair litros do nome do ficheiro (ex: "chuveiro_35.2l.wav")
        match = re.search(r"([\d.]+)l", fname)
        if not match:
            continue
        litros_total = float(match.group(1))

        # Número de chunks completos de 3 segundos
        num_chunks = len(y) // samples_per_chunk
        print(litros_total)
        if num_chunks == 0:
            print("Skipping file: ",fname)
            continue
        elif litros_total != 0:
            litros_por_chunk = litros_total / num_chunks
        else:
            litros_por_chunk = 0

        for i in range(num_chunks):
            start = i * samples_per_chunk
            end = start + samples_per_chunk
            chunk = y[start:end]

            # Extrair MFCCs
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13).T

            # Normalizar
            mfcc = mfcc.astype("float32")
            mfcc = mfcc / np.max(np.abs(mfcc) + 1e-6)

            X_total.append(mfcc)
            y_total.append(litros_por_chunk)

# Padronizar o comprimento das sequências
X_padded = pad_sequences(X_total, maxlen=200, padding='post', dtype='float32')
X = np.array(X_padded)  # shape: (num_samples, 200, 13)
y = np.array(y_total)   # shape: (num_samples,)

# IMPORTANT: Use proper validation split instead of test split for hyperparameter tuning
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Split the temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42  # 15% val, 15% test
)

print(f"✅ Dados carregados de {len(X_total)} chunks")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

def create_improved_model():
    model = models.Sequential([
        # First conv block - smaller and more regularized
        layers.Conv1D(16, 3, activation='relu', input_shape=(200, 13)),  # Reduced from 32
        layers.Dropout(0.3),  # Add dropout early
        layers.MaxPooling1D(2),
        
        # Second conv block - even smaller
        layers.Conv1D(8, 3, activation='relu'),  # Reduced from 16
        layers.BatchNormalization(),
        layers.Dropout(0.4),  # Increase dropout
        layers.MaxPooling1D(2),
        
        # Third conv block - minimal
        layers.Conv1D(4, 1, activation='relu'),   # Reduced from 8
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Global pooling instead of flatten
        layers.GlobalAveragePooling1D(),
        
        # Very small dense layer with heavy dropout
        layers.Dense(4, activation='relu'),  # Reduced from 8
        layers.Dropout(0.6),  # Heavy dropout
        layers.Dense(1)
    ])
    return model



# Use the improved model
model = create_improved_model()

# Compile with lower learning rate
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0003),  # Lower learning rate
    loss='mse', 
    metrics=['mae']
)

# Better callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=8,  # Reduced patience - stop earlier
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# Train with validation split
history = model.fit(
    X_train, y_train, 
    epochs=50,  # Reduced epochs
    validation_data=(X_val, y_val),  # Use validation set
    batch_size=16,  # Increased batch size
    callbacks=callbacks,
    verbose=1
)

model.summary()

# Evaluate on test set (unseen data)
test_mae = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"📊 Test MAE (erro médio absoluto): {test_mae:.4f} litros")
# Evaluate on validation set for comparison
val_mae = model.evaluate(X_val, y_val, verbose=0)[1]
print(f"📊 Validation MAE: {val_mae:.4f} litros")

# Salvar modelo
model.save("models/model_improved.keras")

# Visualização
plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
plt.plot(history.history['mae'], label='MAE Treino', linewidth=2)
plt.plot(history.history['val_mae'], label='MAE Validação', linewidth=2)
plt.title('Erro Médio Absoluto por Época')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Loss Treino', linewidth=2)
# plt.plot(history.history['val_loss'], label='Loss Validação', linewidth=2)
# plt.title('Loss por Época')
# plt.xlabel('Época')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

plt.tight_layout()
plt.savefig("improved_model_training.png", dpi=300)
plt.show()
plt.close()

# Additional analysis
print(f"\n📈 Training History:")
print(f"Final training MAE: {history.history['mae'][-1]:.4f}")
print(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}")
print(f"Best validation MAE: {min(history.history['val_mae']):.4f}")
print(f"Gap between train/val: {abs(history.history['mae'][-1] - history.history['val_mae'][-1]):.4f}")


print("\n🔄 Converting to TensorFlow Lite INT8...")

# Create representative dataset for quantization calibration
def representative_data_gen():
    # Use 100 samples from training data for calibration
    for i in range(min(100, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

# Convert to TFLite with INT8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.representative_dataset = representative_data_gen
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert
try:
    tflite_model = converter.convert()
    print("✅ TFLite conversion successful!")
    
    # Save TFLite model
    tflite_path = "models/model_improved_int8.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Calculate and display sizes
    original_size = os.path.getsize("models/model_improved.keras")
    tflite_size = len(tflite_model)
    
    original_kb = original_size / 1024
    tflite_kb = tflite_size / 1024
    compression_ratio = original_size / tflite_size
    
    print(f"\n📊 Model Size Comparison:")
    print(f"   Original: {original_kb:.2f} KB")
    print(f"   TFLite:   {tflite_kb:.2f} KB")
    print(f"   Compression: {compression_ratio:.1f}x smaller")
    print(f"   Size reduction: {(1-tflite_size/original_size)*100:.1f}%")
    
    # Test the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test with one sample
    test_sample = X_test[0:1]
    input_scale, input_zero_point = input_details[0]['quantization']
    test_quantized = (test_sample / input_scale + input_zero_point).astype(np.int8)
    
    interpreter.set_tensor(input_details[0]['index'], test_quantized)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_scale, output_zero_point = output_details[0]['quantization']
    prediction = (output_data.astype(np.float32) - output_zero_point) * output_scale
    
    print(f"\n🧪 TFLite Test:")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Prediction: {prediction[0][0]:.4f} litros")
    print(f"   Model ready for deployment!")
    
except Exception as e:
    print(f"❌ TFLite conversion failed: {e}")