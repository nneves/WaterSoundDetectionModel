#pragma once

#include "audio_samples.h"

void testWithRealSample(const AudioSample& sample) {
    Serial.printf("\n=== Testing with real sample: %s ===\n", sample.name);
    
    // Extract MFCCs from real audio data
    extractMFCCFeatures(sample.data, sample.length, mfcc_buffer);
    
    // Print MFCC statistics before normalization
    printMFCCStats(mfcc_buffer, N_MFCC * MAX_SEQUENCE_LENGTH, "MFCC (before norm)");
    
    // Normalize features
    normalizeMFCC(mfcc_buffer, N_MFCC * MAX_SEQUENCE_LENGTH);
    
    // Print MFCC statistics after normalization
    printMFCCStats(mfcc_buffer, N_MFCC * MAX_SEQUENCE_LENGTH, "MFCC (after norm)");
    
    // Quantize features for INT8 model
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;
    quantizeMFCC(mfcc_buffer, quantized_input, N_MFCC * MAX_SEQUENCE_LENGTH, 
                 input_scale, input_zero_point);
    
    // Print quantized input statistics
    int8_t min_q = quantized_input[0];
    int8_t max_q = quantized_input[0];
    for (int i = 0; i < N_MFCC * MAX_SEQUENCE_LENGTH; i++) {
        if (quantized_input[i] < min_q) min_q = quantized_input[i];
        if (quantized_input[i] > max_q) max_q = quantized_input[i];
    }
    Serial.printf("Quantized - Min: %d, Max: %d, Scale: %.6f, Zero: %d\n", 
                  min_q, max_q, input_scale, input_zero_point);
    
    // Copy to input tensor and run inference
    memcpy(input->data.int8, quantized_input, N_MFCC * MAX_SEQUENCE_LENGTH * sizeof(int8_t));
    
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status == kTfLiteOk) {
        int8_t prediction_quantized = output->data.int8[0];
        float prediction = (prediction_quantized - output->params.zero_point) * output->params.scale;
        
        float error = fabsf(prediction - sample.expected_liters);
        float error_percent = (error / sample.expected_liters) * 100.0f;
        
        Serial.printf("Prediction: %.4f liters\n", prediction);
        Serial.printf("Expected: %.4f liters\n", sample.expected_liters);
        Serial.printf("Error: %.4f liters (%.1f%%)\n", error, error_percent);
    } else {
        Serial.println("Inference failed!");
    }
    Serial.println("========================\n");
}

void testAllRealSamples() {
    Serial.println("\n=== Testing All Real Audio Samples ===");
    float total_error_percent = 0.0f;
    
    for (int i = 0; i < NUM_AUDIO_SAMPLES; i++) {
        testWithRealSample(audio_samples[i]);
    }
    
    Serial.println("=== Test Complete ===\n");
}
