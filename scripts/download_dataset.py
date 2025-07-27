import os
from roboflow import Roboflow

def download_car_detection_dataset():
    """Download the car detection dataset from Roboflow"""
    
    # Initialize Roboflow
    rf = Roboflow(api_key="H8IuhGEtx3AJRJByOcb4")
    project = rf.workspace("objectdetection-psmvq").project("car-detection-9mzoj-qwddi")
    version = project.version(1)
    
    # Download dataset in COCO format
    print("Downloading car detection dataset...")
    dataset = version.download("coco")
    
    print(f"Dataset downloaded to: {dataset.location}")
    
    # Print dataset structure
    dataset_path = dataset.location
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    return dataset_path

if __name__ == "__main__":
    dataset_path = download_car_detection_dataset()
    print(f"\nDataset ready at: {dataset_path}")
