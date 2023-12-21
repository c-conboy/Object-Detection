# Kitti Dataset Object Detection Inference with ResNet18

This repository contains a Python script for performing object detection inference on the Kitti dataset using a pre-trained ResNet18 model.

## Usage

### Prerequisites

- Python 3
- PyTorch
- OpenCV

### Running the Inference Script

```
python inference.py -i path/to/input_images -idx image_index
```

**Arguments**

-i: Path to the input directory containing Kitti images (default: '../datasets/Kitti8_ROIs/test').

-idx: Image index for inference (default: None).

### Acknowledgments

The script uses a pre-trained ResNet18 model for object detection on the Kitti dataset.



