import numpy as np
import librosa
import os
import sys

def convert_wav_to_header(input_wav, output_header, variable_name):
    """Convert WAV file to C header with float32 samples"""
    # Load and resample audio to 16kHz (matching embedded system)
    y, _ = librosa.load(input_wav, sr=16000)
    
    # Ensure float32 and normalize to [-1, 1]
    y = y.astype(np.float32)
    y = np.clip(y, -1.0, 1.0)
    
    # Create header file content
    with open(output_header, 'w') as f:
        f.write("#pragma once\n\n")
        f.write(f"// Converted from: {os.path.basename(input_wav)}\n")
        f.write(f"// Sample rate: 16000 Hz\n")
        f.write(f"// Duration: {len(y)/16000:.2f} seconds\n")
        f.write(f"// Samples: {len(y)}\n\n")
        
        # Write array length
        f.write(f"const unsigned int {variable_name}_length = {len(y)};\n\n")
        
        # Write sample data
        f.write(f"const float {variable_name}[] = {{\n    ")
        
        # Format numbers in C floating point notation
        samples_str = [f"{x:.8f}f" for x in y]
        
        # Write 8 numbers per line
        for i in range(0, len(samples_str), 8):
            line = ", ".join(samples_str[i:i+8])
            if i + 8 < len(samples_str):
                line += ","
            f.write(line + "\n    ")
        
        f.write("\n};\n")

def main():
    # Test files from your dataset
    test_files = [
        "dataset_sound/audio_samples/1.48l_min_1.wav",
        "dataset_sound/audio_samples/2.60l_1750712105_0.wav",
        "dataset_sound/audio_samples/3.75l_min_1.wav"
    ]
    
    print("Converting audio samples to C headers...")
    
    # Create audio_samples directory if it doesn't exist
    os.makedirs("audio_samples", exist_ok=True)
    
    for wav_file in test_files:
        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        variable_name = f"audio_{base_name.replace('.', '_').replace('-', '_')}"
        header_file = f"audio_samples/{base_name}.h"
        
        print(f"\nProcessing: {wav_file}")
        print(f"Creating: {header_file}")
        convert_wav_to_header(wav_file, header_file, variable_name)
        print(f"Done! Variable name: {variable_name}")

if __name__ == "__main__":
    main()
