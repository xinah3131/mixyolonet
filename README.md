# Mix-YOLONet: Deep Image Restoration for Improving Object Detection

## Overview

Mix-YOLONet is a multi-task joint network designed for both object detection and image restoration. It leverages the YOLOv8 architecture and incorporates attention mechanisms in the encoder and decoder using a Mix Structure Block from MixDehazeNet(https://github.com/AmeryXiong/MixDehazeNet/tree/main).

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

## Dataset
- **VOC-FOG**: Download this model at https://drive.google.com/file/d/1bLUtwrKwzPwLI3yZBFZYw4BnINpxCfVp/view?usp=sharing (from https://github.com/yz-wang/TogetherNet) .
- **RTTS**: Download this model at https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2.
- **Foggy Driving**: Download this model at https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/. 

## Pretrained Model
| Model Name             | Download Link                                  |
|------------------------|-------------------------------------------------|
| Mix-YOLONet | [Link](https://drive.google.com/file/d/1URQIXPIMJMSvucez9HjEglrWu33PrIs5/view?usp=sharing)      |


## Usage

### Training

To train the model, run:
```bash
python train.py
```

### Testing

To test the model, run:
```bash
python train.py --test --data your_dataset
```

### Inference

To perform inference on a single image, run:
```bash
python train.py --inference --image_path path_to_your_image
```

To perform inference on an entire folder of images, run:
```bash
python train.py --inference_test --model_path your_model --inference_input your_input_image
```

### Data Conversion

Convert VOCFOG dataset:
```bash
python voctoyolo.py 
```

Convert Foggy Driving dataset:
```bash
python foggytovoc.py 
```

Convert RTTS dataset:
```bash
python rttstovoc.py 
```

## Configuration

Modify the `configs/config.yaml` file to change training parameters, paths, and other settings.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- YOLOv8 by jahongir7174 (https://github.com/jahongir7174/YOLOv8-pt)
- Mix DehazeNet for providing the mix structure block by AmeryXiong (https://github.com/AmeryXiong/MixDehazeNet/tree/main)

## Contributing

Feel free to submit issues and pull requests for improvements or bug fixes.
```