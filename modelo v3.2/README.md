# Modelo v3.2

Machine Learning Model v3.2 for Audio Classification using TensorFlow and Keras.

## Overview

This project contains an improved audio classification model that can distinguish between different types of sounds. The model is built using TensorFlow/Keras and includes audio processing capabilities with librosa.

## Features

- Audio classification using deep learning
- Support for various audio formats
- Real-time audio processing
- TensorFlow Lite model conversion for deployment
- ESP32 compatibility for edge deployment

## Installation

See [README_UV_SETUP.md](README_UV_SETUP.md) for detailed installation instructions using UV.

### Quick Start with UV

```bash
# Install UV
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install dependencies
uv sync

# Activate environment
.venv\Scripts\activate
```

## Usage

```python
# Example usage
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model_improved_3_2_1.keras')

# Make predictions
# Your audio processing code here
```

## Files

- `model_improved_3_2_1.keras` - Main Keras model
- `model_improved_int8_3_2_1.tflite` - TensorFlow Lite quantized model
- `predict.py` - Prediction script
- `predict_esp32.py` - ESP32-compatible prediction script
- `audio_samples_3_2/` - Training audio samples
- `audio_samples_3_2_1/` - Additional training samples

## Requirements

- Python 3.10 or higher
- TensorFlow 2.18.0
- Librosa for audio processing
- NumPy, Pandas for data handling

## License

MIT License
