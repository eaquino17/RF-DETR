# RF-DETR Object Detection Project

This project implements a complete RF-DETR (Receptive Field Enhanced Detection Transformer) system for object detection using a car detection dataset from Roboflow.

## Project Overview

The RF-DETR model enhances the original DETR (Detection Transformer) architecture by incorporating receptive field enhancement modules that improve the model's ability to capture multi-scale features for better object detection performance.

## Features

- **Complete RF-DETR Implementation**: Full implementation with CNN backbone, Transformer encoder-decoder, and RF enhancement modules
- **Data Pipeline**: Automated dataset download and preprocessing with COCO format support
- **Training Pipeline**: Comprehensive training with loss functions, optimization, and checkpointing
- **Evaluation Metrics**: Implementation of mAP, precision, recall, and IoU metrics
- **Visualization**: Prediction visualization and performance plotting

## Project Structure

\`\`\`
rf-detr-project/
├── src/
│   ├── data_loader.py          # Dataset loading and preprocessing
│   ├── rf_detr_model.py        # RF-DETR model implementation
│   ├── hungarian_matcher.py    # Hungarian matching algorithm
│   ├── loss.py                 # Loss functions for training
│   ├── trainer.py              # Training pipeline
│   ├── evaluator.py            # Evaluation and metrics
│   └── utils.py                # Utility functions
├── scripts/
│   ├── install_dependencies.py # Dependency installation
│   ├── download_dataset.py     # Dataset download
│   ├── main_training.py        # Main training script
│   └── run_project.py          # Complete pipeline runner
└── README.md
\`\`\`

## Quick Start

### Option 1: Run Complete Pipeline
```python
python scripts/run_project.py
