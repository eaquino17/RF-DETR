RT-DETR Object Detection on CIFAR-10 Dataset
This project demonstrates an end-to-end implementation of the RT-DETR (Real-Time Detection Transformer) model applied to the CIFAR-10 dataset. It showcases how a classification dataset can be adapted for object detection using a cutting-edge transformer-based architecture.

Authors: Lester Dave C. Ablat and Eric Bernard Aquino
Course: CSS182-4 – CO2.2

🎯 Objectives
Convert CIFAR-10 (originally for classification) to an object detection dataset.

Train and evaluate the RT-DETR model on this custom dataset.

Analyze model performance using evaluation metrics and visual tools.

🏗️ Project Workflow
1. Install Dependencies
Install the necessary packages:

pip install -r requirements.txt

Essential dependencies include:

torch

torchvision

opencv-python

matplotlib

ultralytics

pyyaml

tqdm

2. Dataset Preparation
CIFAR-10 is automatically downloaded and upscaled from 32x32 to 640x640.

Bounding boxes are synthetically created per class and exported in YOLO format.

Dataset is split into training and validation subsets.

3. Model Loading
Supports loading of rtdetr-s.pt, rtdetr-m.pt, and rtdetr-l.pt weights.

Includes fallback if weights are not found.

4. Training
Model is trained using detection-optimized hyperparameters (batch size, epochs, learning rate).

Losses: giou_loss, cls_loss, l1_loss.

5. Evaluation
Metrics: mAP50, mAP50-95, precision, recall, F1-score.

Visuals: Loss curves, confusion matrix, PR curve, and F1-confidence curve.

6. Visualization
Predicted bounding boxes are overlaid on test images.

Color-coded bounding boxes for correct (green) and incorrect (red) predictions.

📊 Results Summary
Losses
train/giou: Gradually decreased post-epoch 5, stabilizing at lower values.

train/cls: Steady decline, indicating improving classification accuracy.

train/l1: Fluctuated but improved in late epochs, signaling better box regression.

val/*: Higher/noisier — common for small datasets like CIFAR-10.

Evaluation Metrics
Precision: ~0.75 - 0.8

Recall: ~0.75 - 0.8

mAP@0.5: ~0.60+

mAP@0.5:0.95: ~0.40

F1-score: Max ~0.76

Confusion Matrix Insights
High true positives for horse, truck, and frog.

Frequent misclassifications for cat, dog, bird due to visual similarity.

Background errors (false positives/negatives) affected recall/precision.

Normalized matrix revealed class-specific weaknesses (e.g., cat → bird: 14%).

🚧 Limitations
CIFAR-10’s small resolution limited detection quality.

Synthetic bounding boxes may not capture real-world variance.

Confidence scores showed poor calibration (best F1 @ 0.000 threshold).

RT-DETR’s anchor-free nature struggles with small-object localization.

🔗 Resources & References
https://docs.ultralytics.com/models/rtdetr/

https://blog.roboflow.com/what-is-a-confusion-matrix/

https://docs.ultralytics.com/integrations/albumentations/

https://medium.com/@k.sunman91/data-augmentation-on-ultralytics-for-training-yolov5-yolov8-97a8dab31fef

https://arxiv.org/abs/2304.08069

https://arxiv.org/abs/2005.12872

https://www.researchgate.net/figure/F1-confidence-curve-of-the-model_fig4_378729812

https://www.researchgate.net/figure/The-precision-confidence-curve_fig5_374492535

🗂️ Repository Structure
.
├── main.py # Full pipeline: data prep, model loading, training, eval
├── requirements.txt # Required dependencies
├── README.md # Project documentation
└── data/ # (Optional) Custom YOLO-formatted CIFAR-10 dataset

📥 Dataset & Checkpoints
CIFAR-10 is auto-downloaded. RT-DETR checkpoints must be placed in your working directory or handled via fallback.

You can also access shared project assets:
Google Drive Link: https://drive.google.com/drive/folders/1wdZ6JYSaqwIym6IlUwxo1LT7toFUzkzM?usp=sharing

✍️ Authors
Lester Dave C. Ablat

Eric Bernard Aquino

