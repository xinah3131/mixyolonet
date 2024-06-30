Sure, here is the complete `README.md` for your project:

```markdown
# MixYoloNet: Deep Image Restoration for Improving Object Detection

## Overview

MixYoloNet is a multi-task joint network designed for both object detection and image restoration. It leverages the YOLOv8 architecture and incorporates attention mechanisms in the encoder and decoder using a Mix Structure Block from Mix Structure Net.

## Features

- **Training**: Train the model using `train.py`.
- **Testing**: Test the model's performance using `train.py --test`.
- **Inference**: Perform inference on a single image or an entire folder of images.
- **Data Conversion**: Convert various datasets (VOC, Foggy Driving, RTTS) to the required format.

## Installation

Clone this repository:
```bash
git clone https://github.com/yourusername/MixYoloNet.git
cd MixYoloNet
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model, run:
```bash
python train.py
```

### Testing

To test the model, run:
```bash
python train.py --test
```

### Inference

To perform inference on a single image, run:
```bash
python train.py --inference --image_path path_to_your_image
```

To perform inference on an entire folder of images, run:
```bash
python train.py --inference_test --folder_path path_to_your_folder
```

### Data Conversion

Convert VOCFOG dataset:
```bash
python voctoyolo.py --dataset_path path_to_vocfog_dataset
```

Convert Foggy Driving dataset:
```bash
python foggytovoc.py --dataset_path path_to_foggy_driving_dataset
```

Convert RTTS dataset:
```bash
python rttstovoc.py --dataset_path path_to_rtts_dataset
```

## Configuration

Modify the `configs/config.yaml` file to change training parameters, paths, and other settings.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- YOLOv8 by Ultralytics
- Mix Structure Net for providing the mix structure block

## Contributing

Feel free to submit issues and pull requests for improvements or bug fixes.
```

You can copy and paste this `README.md` into your GitHub repository. Make sure to adjust any specific paths, commands, or configurations as needed for your project.