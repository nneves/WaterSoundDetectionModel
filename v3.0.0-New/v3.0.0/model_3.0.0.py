import os
import re
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Diretório com os ficheiros de áudio (.wav, .mp3, etc.)
audio_dir = "dataset_sound/audio_samples"
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
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20).T

            # Normalizar
            mfcc = mfcc.astype("float32")
            mfcc = mfcc / np.max(np.abs(mfcc) + 1e-6)

            X_total.append(mfcc)
            y_total.append(litros_por_chunk)

# Padronizar o comprimento das sequências
X_padded = pad_sequences(X_total, maxlen=200, padding='post', dtype='float32')
X = np.array(X_padded)  # shape: (num_samples, 200, 20)
y = np.array(y_total)   # shape: (num_samples,)

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Dados carregados de {len(X_total)} chunks")
print(f"Shape X: {X.shape}, Shape y: {y.shape}")

# Modelo pequeno para uso em embutidos
model = models.Sequential([
    layers.Conv1D(64, 3, activation='relu', input_shape=(200, 20)),
    layers.MaxPooling1D(2),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(16, 1, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=8)

model.summary()

# Salvar modelo
model.save("model_3.0.0.keras")

# Avaliação
mae = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"📊 MAE (erro médio absoluto): {mae:.2f} litros")

# Visualização
plt.plot(history.history['mae'], label='MAE Treino')
plt.plot(history.history['val_mae'], label='MAE Validação')
plt.title('Erro Médio Absoluto por Época')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.show()
