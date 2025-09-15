# UV Setup for Modelo v3.0.0 (1.8% Average Error)

This project uses UV, a fast Python package manager, for dependency management. Our tests show an average error of 1.8% across different water flow measurements.

## Requirements

- **Python 3.10 or higher** (required by dependencies like tensorflow, librosa, etc.)

## Installation and Setup

### 1. Install UV
```bash
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create and Activate Virtual Environment

```bash
# Navigate to the project directory
cd "ModeloAI/v3.0.0-New/v3.0.0"

# Create virtual environment
uv venv

# Activate the virtual environment
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate

# Install dependencies using UV
uv pip install -r requirements.txt
```

## Project Structure

- `model_3.0.0.keras` - Trained model with 1.8% average error
- `model_3.0.0.tflite` - TFLite version of the model
- `predict.py` - Script for analyzing audio files (1.8% avg error)
- `script_predict.py` - Real-time prediction script
- `requirements.txt` - Project dependencies
- `dataset_sound/` - Test audio samples
- `model_improvements.md` - Documentation of potential improvements

## Usage Commands

### Running Predictions
```bash
# Activate virtual environment (if not already activated)
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Run file-based predictions
uv run predict.py

# Run real-time predictions
uv run script_predict.py
```

### Package Management
```bash
# Install all dependencies
uv pip install -r requirements.txt

# Add a new dependency
uv pip install package-name

# Update all dependencies
uv pip install --upgrade -r requirements.txt
```

## Performance Metrics

- Average Error: 1.8%
- Best Case: 0.4% error (2.60L sample)
- Worst Case: 4.0% error (1.48L sample)
- Consistent sub-4% error across all test cases

## Benefits of UV

- **Fast**: 10-100x faster than pip
- **Reliable**: Consistent dependency resolution
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Simple**: Direct requirements.txt support

## Notes

- The model processes audio in 3-second chunks
- Each prediction includes detailed per-chunk analysis
- The system maintains accuracy across different flow rates (1.48L to 3.75L)
- See `model_improvements.md` for future enhancement plans