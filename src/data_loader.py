import torch
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import numpy as np
from PIL import Image
import os
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, max_objects=100):
        self.root_dir = root_dir
        self.transform = transform
        self.max_objects = max_objects
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        # Filter images that have annotations
        self.image_ids = list(self.image_annotations.keys())
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        annotations = self.image_annotations[img_id]
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        
        for ann in annotations:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Normalize boxes to [0, 1]
        h, w = image.shape[:2]
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Pad or truncate to max_objects
        num_objects = len(boxes)
        if num_objects > self.max_objects:
            boxes = boxes[:self.max_objects]
            labels = labels[:self.max_objects]
            num_objects = self.max_objects
        elif num_objects < self.max_objects:
            # Pad with zeros
            pad_boxes = torch.zeros((self.max_objects - num_objects, 4))
            pad_labels = torch.zeros((self.max_objects - num_objects,), dtype=torch.long)
            boxes = torch.cat([boxes, pad_boxes])
            labels = torch.cat([labels, pad_labels])
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'num_objects': num_objects
        }

def get_transforms(is_train=True):
    """Get data transforms for training and validation"""
    if is_train:
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def create_data_loaders(dataset_path, batch_size=4):
    """Create train, validation, and test data loaders"""
    
    # Define paths
    train_images = os.path.join(dataset_path, 'train')
    val_images = os.path.join(dataset_path, 'valid')
    test_images = os.path.join(dataset_path, 'test')
    
    train_ann = os.path.join(dataset_path, 'train', '_annotations.coco.json')
    val_ann = os.path.join(dataset_path, 'valid', '_annotations.coco.json')
    test_ann = os.path.join(dataset_path, 'test', '_annotations.coco.json')
    
    # Create datasets
    train_dataset = COCODataset(train_images, train_ann, get_transforms(True))
    val_dataset = COCODataset(val_images, val_ann, get_transforms(False))
    test_dataset = COCODataset(test_images, test_ann, get_transforms(False))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader, train_dataset.categories

if __name__ == "__main__":
    # Test data loading
    dataset_path = "path/to/your/dataset"  # Update this path
    train_loader, val_loader, test_loader, categories = create_data_loaders(dataset_path)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Categories: {categories}")
    
    # Test loading a batch
    for batch in train_loader:
        print(f"Image shape: {batch['image'].shape}")
        print(f"Boxes shape: {batch['boxes'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        break
