#pragma once

// Include all audio sample headers
#include "audio_samples/1_48l_min_1.h"
#include "audio_samples/2_60l_1750712105_0.h"
#include "audio_samples/3_75l_min_1.h"

// Audio sample struct for easy access
struct AudioSample {
    const float* data;
    const unsigned int length;
    const float expected_liters;
    const char* name;
};

// Array of all available samples
const AudioSample audio_samples[] = {
    {
        audio_1_48l_min_1,
        audio_1_48l_min_1_length,
        1.48f,
        "Low flow (1.48L)"
    },
    {
        audio_2_60l_1750712105_0,
        audio_2_60l_1750712105_0_length,
        2.60f,
        "Medium flow (2.60L)"
    },
    {
        audio_3_75l_min_1,
        audio_3_75l_min_1_length,
        3.75f,
        "High flow (3.75L)"
    }
};

const int NUM_AUDIO_SAMPLES = sizeof(audio_samples) / sizeof(audio_samples[0]);
