# RT-DETR Object Detection on CIFAR-10 Dataset

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-rtdetr-%2300A3FF)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Project Banner](assets/banner.png)

## Table of Contents
- [Project Objectives](#project-objectives)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Limitations](#limitations)
- [Repository Structure](#repository-structure)
- [Resources](#resources)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Authors](#authors)

## Project Objectives
- Convert CIFAR-10 classification dataset to object detection format
- Train and evaluate RT-DETR model on custom dataset
- Analyze performance using evaluation metrics and visual tools

## Installation

# Clone repository
git clone https://github.com/yourusername/rtdetr-cifar10.git
cd rtdetr-cifar10

# Install dependencies
pip install -r requirements.txt

Required packages:

torch>=2.0.0

torchvision>=0.15.0

opencv-python>=4.7.0

ultralytics>=8.0.0

pyyaml>=6.0

matplotlib>=3.7.0

Dataset Preparation
CIFAR-10 is automatically downloaded

Images upscaled from 32x32 to 640x640

Synthetic bounding boxes generated per class in YOLO format:
<class_id> <x_center> <y_center> <width> <height>

## Model Training

# Train with default parameters
python train.py --model rtdetr-s --epochs 50 --batch 16

# Available models: rtdetr-s, rtdetr-m, rtdetr-l

Training configuration:

Optimizer: AdamW

Learning rate: 0.0001

Loss functions: GIoU, Classification, L1

Augmentations: Random flip, color jitter

## Evaluation

Metrics calculated on validation set:

python
from ultralytics import RTDETR

model = RTDETR('runs/train/weights/best.pt')
metrics = model.val(data='cifar10.yaml')
Key evaluation metrics:

mAP@0.5: 0.62

mAP@0.5:0.95: 0.41

Precision: 0.78

Recall: 0.76

F1-score: 0.77

## Authors
# Eric Bernard Aquino
# Lester Dave C. Ablat


Course: CSS182-4 – CO2.2

