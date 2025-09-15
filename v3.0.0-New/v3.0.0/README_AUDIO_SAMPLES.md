# Converting Audio Samples for Embedded Testing

This guide explains how to convert WAV audio samples into C header files for use in the ESP32 embedded system. This allows testing the TFLite model with the exact same audio data used in the Python development environment.

## Project Structure

```
├── convert_audio_samples.py    # Conversion script
├── audio_samples.h            # Main header including all samples
├── audio_samples/             # Directory containing converted samples
│   ├── 1_48l_min_1.h         # Low flow sample
│   ├── 2_60l_1750712105_0.h  # Medium flow sample
│   └── 3_75l_min_1.h         # High flow sample
├── test_real_samples.h        # ESP32 testing functions
└── uv-requirements.txt        # UV-compatible requirements file
```

## Setting Up the Environment

### 1. Install UV

```bash
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
uv venv

# Activate virtual environment
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install using UV (faster and more reliable)
uv pip install -r uv-requirements.txt
```

This will install all required packages including:
- librosa==0.11.0
- numpy==1.26.4
- tensorflow==2.18.0
- and other dependencies

## Converting Audio Samples

1. **Ensure Virtual Environment is Activated:**
   ```bash
   # Check if UV environment is active
   # Your prompt should show (.venv)
   ```

2. **Run the conversion script:**
   ```bash
   uv run python convert_audio_samples.py
   ```
   This will:
   - Load each WAV file
   - Resample to 16kHz (matching embedded system)
   - Normalize to [-1, 1] range
   - Generate C header files in `audio_samples/`

3. **Generated Files:**
   - Each WAV file generates a corresponding `.h` file
   - Files contain:
     - Float array of audio samples
     - Array length
     - Metadata (original filename, duration, etc.)

## Memory Usage

Each audio sample requires:
- 3 seconds × 16000 Hz = 48000 samples
- Each sample is float32 (4 bytes)
- Total per sample ≈ 192KB
- Three test files ≈ 576KB total

Ensure your ESP32 has sufficient program memory (flash) available.

## Integration with ESP32 Code

1. **Copy Generated Files:**
   - Copy `audio_samples.h`, `test_real_samples.h`, and the `audio_samples/` directory to your ESP32 project
   - Place them in the same directory as your `main.cpp`

2. **Include Headers:**
   ```cpp
   #include "audio_samples.h"
   #include "test_real_samples.h"
   ```

3. **Add Command Handler:**
   ```cpp
   // In your loop() command handler:
   else if (inputValue == "r" || inputValue == "R") {
       testAllRealSamples();
   }
   ```

4. **Add Help Message:**
   ```cpp
   // In your showHelp() function:
   Serial.println("'r' or 'R' - Test with real audio samples");
   ```

## Testing Real Samples

1. **Test All Samples:**
   - Send 'r' command through Serial monitor
   - System will test all samples sequentially
   - Results show MFCC stats and predictions

2. **Test Individual Sample:**
   ```cpp
   // Example in your code:
   testWithRealSample(audio_samples[0]); // Test low flow
   ```

## Output Format

The test output matches the Python analysis format:
```
=== Testing with real sample: Low flow (1.48L) ===
MFCC (before norm) - Min: -801.3386, Max: 61.3263, Mean: -12.8542, Range: 862.6649
MFCC (after norm) - Min: -1.0000, Max: 0.0765, Mean: -0.0160, Range: 1.0765
Quantized - Min: -1, Max: 127, Scale: 0.004222, Zero: -1
Prediction: 1.54 liters
Expected: 1.48 liters
Error: 0.06 liters (4.0%)
```

## UV Package Management Tips

1. **Adding New Dependencies:**
   ```bash
   uv pip install package_name
   # Update requirements file
   uv pip freeze > uv-requirements.txt
   ```

2. **Updating Dependencies:**
   ```bash
   uv pip install --upgrade -r uv-requirements.txt
   ```

3. **Checking Environment:**
   ```bash
   uv pip list
   ```

4. **Clean Installation:**
   ```bash
   # Remove virtual environment
   rm -rf .venv
   # Create fresh environment
   uv venv
   # Reinstall dependencies
   uv pip install -r uv-requirements.txt
   ```

## Troubleshooting

1. **UV Environment Issues:**
   - Ensure UV is properly installed: `uv --version`
   - Check virtual environment activation
   - Try clean installation if dependencies conflict

2. **Memory Issues:**
   - If you get allocation failures, check:
     - Available program memory
     - Tensor arena size
     - Stack size settings

3. **Accuracy Differences:**
   - Compare MFCC statistics with Python output
   - Verify quantization parameters
   - Check normalization ranges

4. **Performance Issues:**
   - Monitor processing time
   - Consider reducing sample count
   - Check CPU frequency settings

## Notes

- Audio data is stored in program memory (flash)
- No file system or SD card required
- Same preprocessing pipeline as live audio
- Exact reproducibility with Python results
- Useful for validation and debugging

## Future Improvements

1. Add compression for audio data
2. Include more test samples
3. Add batch testing mode
4. Implement memory usage optimization
5. Add performance benchmarking

## UV Benefits

- 10-100x faster than pip
- Reliable dependency resolution
- Consistent environments across platforms
- Better handling of pre-built wheels
- Improved caching and download speeds