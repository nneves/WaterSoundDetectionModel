# Converting TFLite Model to C Header File

This guide explains how to convert the `model_3.0.0.tflite` model into a C header file for embedded systems using the `xxd` tool in a Docker container.

## Prerequisites

- Docker installed on your system
- The `model_3.0.0.tflite` file in your project directory

## Setup Docker Environment

1. First, create a Dockerfile:

```dockerfile
FROM ubuntu:latest

# Install xxd (part of vim-common)
RUN apt-get update && apt-get install -y vim-common

# Set working directory
WORKDIR /data

# Keep container running
ENTRYPOINT ["/bin/bash"]
```

2. Save this as `Dockerfile` in your project directory.

## Converting the Model

1. **Build the Docker container:**
```bash
docker build -t xxd-container .
```

2. **Run the container with volume mounting:**
```bash
# On Windows PowerShell
docker run -it --rm -v ${PWD}:/data xxd-container

# On Windows CMD
docker run -it --rm -v %cd%:/data xxd-container

# On Linux/macOS
docker run -it --rm -v $(pwd):/data xxd-container
```

3. **Inside the container, convert the TFLite model:**
```bash
# Convert the model to a C header file
xxd -i model_3.0.0.tflite > model_3_0_0.h
```

The generated header file will contain:
- An array containing the model data
- A variable with the size of the array

4. **Optional: Clean up the header file:**
```bash
# Add const and alignment (run these commands in the container)
sed -i '1i\#pragma once\n' model_3_0_0.h
sed -i 's/unsigned char/const unsigned char __attribute__((aligned(8)))/' model_3_0_0.h
```

## Output

The generated `model_3_0_0.h` file will look something like this:

```c
#pragma once

const unsigned char __attribute__((aligned(8))) model_3_0_0_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...
};
unsigned int model_3_0_0_tflite_len = 123456;
```

## Using the Header File

1. Copy the generated `model_3_0_0.h` file to your embedded project
2. Include it in your C/C++ code:
```c
#include "model_3_0_0.h"

// The model data is available as:
// - model_3_0_0_tflite[] (array containing model data)
// - model_3_0_0_tflite_len (size of the array)
```

## Important Notes

- The generated header file might be large depending on your model size
- Make sure your embedded system has enough program memory to store the model
- The `__attribute__((aligned(8)))` ensures proper memory alignment for most platforms
- The model array is marked as `const` to store it in program memory rather than RAM

## Troubleshooting

1. **File not found in container:**
   - Make sure the model file is in your project directory
   - Verify the volume mounting path

2. **Permission issues:**
   - Ensure you have write permissions in the current directory
   - Try running Docker with appropriate permissions

3. **Large file handling:**
   - For very large models, consider using compression
   - Some embedded systems might need the model split into smaller chunks

## Next Steps

1. Test the generated header file in your embedded project
2. Consider implementing model compression if size is an issue
3. Verify memory alignment requirements for your specific platform
