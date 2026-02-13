# LET-NET Python Inference

Python implementation of LET-NET inference using ONNX Runtime. This is a port of the original C++ code that uses ncnn to Python using onnxruntime.

## Features

- Pure Python implementation with minimal dependencies
- Uses ONNX models for inference (no ncnn required)
- Preserves all original logic from C++ implementation
- Corner tracking with optical flow
- Grid-based feature extraction
- Support for both video and image pair inputs

## Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- numpy>=1.21.0
- opencv-python>=4.5.0
- onnxruntime>=1.12.0

## Usage

### Process a video file

```bash
python main.py <path_to_onnx_model> <path_to_video>
```

Example:
```bash
python main.py model/letnet.onnx assets/nyu_snippet.mp4
```

### Process two images

```bash
python main.py <path_to_onnx_model> <image1> <image2>
```

Example:
```bash
python main.py model/letnet.onnx assets/1.png assets/2.png
```

### Headless mode (no display, save output to files)

```bash
python main.py <path_to_onnx_model> <input> --no-display --output-dir output
```

This is useful for running on servers without display or in automated pipelines.

## Available ONNX Models

The repository includes two ONNX models in the `model/` directory:

- `letnet.onnx` - Standard RGB model
- `letnet-gray.onnx` - Grayscale model

## Implementation Details

### Preprocessing

1. Images are resized to 320x240 pixels
2. Normalized to [0, 1] by dividing by 255
3. Converted from BGR to RGB
4. Transposed to CHW (channels, height, width) format
5. H and W are swapped to match ONNX model expectations

### Model Inference

The model outputs two tensors:
- **Score map** (1, 1, 320, 240): Confidence scores for corner detection
- **Descriptor map** (1, 3, 320, 240): Feature descriptors for tracking

### Postprocessing

1. Dimensions are swapped back from model output
2. Values are denormalized to [0, 255]
3. Converted from RGB back to BGR for OpenCV

### Corner Tracking

The tracking module implements:
- Grid-based feature extraction with non-maximum suppression
- Lucas-Kanade optical flow tracking
- Trajectory history (up to 5 frames)
- Automatic feature replenishment

## Performance

On a typical CPU:
- Preprocessing: ~0.5ms per frame
- Inference: ~2-5ms per frame
- Postprocessing: ~0.8ms per frame
- Total: ~3-6ms per frame (160-330 FPS)

## Differences from C++ Implementation

1. **Framework**: Uses onnxruntime instead of ncnn
2. **Model format**: Uses .onnx files instead of .param/.bin files
3. **Language**: Pure Python instead of C++
4. **Display**: Added --no-display option for headless execution
5. **Output**: Saves individual frames as PNG files in video mode

## Code Structure

- `main.py` - Main inference script
- `tracking.py` - Corner tracking implementation
- `requirements.txt` - Python dependencies

## Notes

- The ONNX model has H/W dimensions swapped compared to the typical PyTorch convention
- All original logic from the C++ implementation is preserved
- The tracking algorithm is identical to the C++ version
- Feature extraction uses the same grid-based approach with the same parameters
