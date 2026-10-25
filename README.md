# Butterfly Species Classification using PyTorch

A computer vision project that classifies butterfly species using a custom Convolutional Neural Network (CNN) with Channel and Spatial Attention, compared against a pretrained ResNet18 model.

## Overview

This project explores image classification from scratch and transfer learning on the Butterfly Images — 40 Species dataset from Kaggle.

The custom CNN learns visual features directly from the dataset. Channel Attention emphasizes informative feature maps, while Spatial Attention helps the model focus on relevant image regions.

A pretrained ResNet18 provides a transfer-learning baseline for comparison.

## Models

### 1. Custom CNN with Attention

- Five convolutional blocks
- Channel Attention and Spatial Attention
- Batch normalization and ReLU activations
- Dropout regularization
- AdamW optimizer and cosine learning-rate scheduling
- Mixed-precision GPU training

### 2. Pretrained ResNet18

- ImageNet-pretrained weights
- New classification head for the butterfly species
- Initial classifier training followed by fine-tuning of the final residual block

## Results

| Metric | Custom CNN + Attention | Pretrained ResNet18 |
|---|---:|---:|
| Test accuracy | 97% | See `results/resnet18_test_results.json` |
| Macro F1 | See `results/test_results.json` | See `results/resnet18_test_results.json` |
| Weighted F1 | See `results/test_results.json` | See `results/resnet18_test_results.json` |

The custom model achieved 97% accuracy on the 500-image test set. ResNet18 results are reported separately for comparison.

## Explainability

Grad-CAM visualizations help inspect the image regions that influence the custom CNN's predictions.

See `results/gradcam_examples.png`.

Grad-CAM provides an approximate visual explanation, not proof of which features the model uses.

## Dataset

Dataset: [Butterfly Images — 40 Species (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species)

Download the dataset from Kaggle and arrange it into the expected training, validation, and test folders.

```text
data/
├── train/
│   └── species_name/
├── valid/
│   └── species_name/
└── test/
    └── species_name/
```

The dataset is not included in this repository.

## Project Structure

```text
classification/
├── preprocess.py
├── train.py
├── train_pretrained.py
└── evaluate.py
models/
├── best_model.pth
└── best_resnet18.pth
results/
├── training_curves.png
├── confusion_matrix.png
├── gradcam_examples.png
├── classification_report.txt
└── test_results.json
requirements.txt
README.md
```

## Installation

Python 3.10 or newer is recommended.

Create and activate a virtual environment, then install PyTorch with a CUDA-enabled build compatible with your NVIDIA driver if GPU acceleration is available.

```bash
pip install -r requirements.txt
```

## Training

Train the custom attention CNN:

```bash
python classification/train.py
```

Train and evaluate the pretrained ResNet18:

```bash
python classification/train_pretrained.py
```

Evaluate the custom CNN:

```bash
python classification/evaluate.py
```

The dataset paths and image preprocessing settings must match the configuration in `classification/preprocess.py`.

## Hardware

Developed and trained on a Windows laptop with an NVIDIA GeForce RTX 3050 Laptop GPU with 6 GB VRAM.

## Limitations

- Accuracy depends on the dataset, class distribution, and test split.
- Results may not generalize to butterfly images collected under different conditions.
- Grad-CAM is an approximate explanation technique.
- Model comparisons are meaningful only when evaluated on the same held-out test split.

## License

Code and dataset licensing should be reviewed separately. The Kaggle dataset's license and the pretrained model's license determine the applicable redistribution conditions.