import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
import cv2
from src.utils import box_cxcywh_to_xyxy, box_iou

class RFDETREvaluator:
    def __init__(self, model, test_loader, device, categories, iou_threshold=0.5):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.categories = categories
        self.iou_threshold = iou_threshold
        
    def compute_metrics(self, predictions, targets):
        """Compute precision, recall, and mAP"""
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_gt_boxes = []
        all_gt_labels = []
        
        for pred, target in zip(predictions, targets):
            # Get predictions above confidence threshold
            scores = pred['scores']
            boxes = pred['boxes']
            labels = pred['labels']
            
            # Filter by confidence
            keep = scores > 0.5
            scores = scores[keep]
            boxes = boxes[keep]
            labels = labels[keep]
            
            all_pred_boxes.append(boxes)
            all_pred_scores.append(scores)
            all_pred_labels.append(labels)
            all_gt_boxes.append(target['boxes'])
            all_gt_labels.append(target['labels'])
        
        # Compute mAP
        ap_scores = []
        for class_id in range(len(self.categories)):
            # Get predictions and targets for this class
            class_pred_boxes = []
            class_pred_scores = []
            class_gt_boxes = []
            
            for i in range(len(all_pred_boxes)):
                # Predictions for this class
                class_mask = all_pred_labels[i] == class_id
                if class_mask.any():
                    class_pred_boxes.extend(all_pred_boxes[i][class_mask])
                    class_pred_scores.extend(all_pred_scores[i][class_mask])
                
                # Ground truth for this class
                gt_mask = all_gt_labels[i] == class_id
                if gt_mask.any():
                    class_gt_boxes.extend(all_gt_boxes[i][gt_mask])
            
            if len(class_pred_scores) == 0 or len(class_gt_boxes) == 0:
                ap_scores.append(0.0)
                continue
            
            # Convert to tensors
            class_pred_boxes = torch.stack(class_pred_boxes) if class_pred_boxes else torch.empty(0, 4)
            class_pred_scores = torch.tensor(class_pred_scores) if class_pred_scores else torch.empty(0)
            class_gt_boxes = torch.stack(class_gt_boxes) if class_gt_boxes else torch.empty(0, 4)
            
            # Compute AP for this class
            ap = self.compute_ap(class_pred_boxes, class_pred_scores, class_gt_boxes)
            ap_scores.append(ap)
        
        mean_ap = np.mean(ap_scores)
        
        # Compute overall precision and recall
        precision, recall = self.compute_precision_recall(all_pred_boxes, all_pred_scores, 
                                                         all_gt_boxes, all_gt_labels)
        
        return {
            'mAP': mean_ap,
            'AP_per_class': ap_scores,
            'precision': precision,
            'recall': recall
        }
    
    def compute_ap(self, pred_boxes, pred_scores, gt_boxes):
        """Compute Average Precision for a single class"""
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return 0.0
        
        # Sort predictions by score
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        
        # Compute IoU between all predictions and ground truth
        ious = box_iou(pred_boxes, gt_boxes)
        
        # Match predictions to ground truth
        tp = torch.zeros(len(pred_boxes))
        fp = torch.zeros(len(pred_boxes))
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
        
        for i in range(len(pred_boxes)):
            # Find best matching ground truth
            if len(gt_boxes) > 0:
                max_iou, max_idx = torch.max(ious[i], dim=0)
                if max_iou >= self.iou_threshold and not gt_matched[max_idx]:
                    tp[i] = 1
                    gt_matched[max_idx] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / len(gt_boxes)
        
        # Compute AP using 11-point interpolation
        ap = 0
        for t in torch.arange(0, 1.1, 0.1):
            if torch.sum(recall >= t) == 0:
                p = 0
            else:
                p = torch.max(precision[recall >= t])
            ap += p / 11
        
        return ap.item()
    
    def compute_precision_recall(self, all_pred_boxes, all_pred_scores, all_gt_boxes, all_gt_labels):
        """Compute overall precision and recall"""
        total_tp = 0
        total_fp = 0
        total_gt = 0
        
        for i in range(len(all_pred_boxes)):
            pred_boxes = all_pred_boxes[i]
            pred_scores = all_pred_scores[i]
            gt_boxes = all_gt_boxes[i]
            
            if len(pred_boxes) == 0:
                total_gt += len(gt_boxes)
                continue
            
            if len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue
            
            # Compute IoU
            ious = box_iou(pred_boxes, gt_boxes)
            
            # Match predictions to ground truth
            gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
            
            for j in range(len(pred_boxes)):
                max_iou, max_idx = torch.max(ious[j], dim=0)
                if max_iou >= self.iou_threshold and not gt_matched[max_idx]:
                    total_tp += 1
                    gt_matched[max_idx] = True
                else:
                    total_fp += 1
            
            total_gt += len(gt_boxes)
        
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_gt + 1e-8)
        
        return precision, recall
    
    def evaluate(self):
        """Evaluate the model on test set"""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                
                # Get model predictions
                outputs = self.model(images)
                
                # Process predictions
                for i in range(len(images)):
                    pred_logits = outputs['pred_logits'][i]
                    pred_boxes = outputs['pred_boxes'][i]
                    
                    # Convert logits to probabilities
                    pred_probs = torch.softmax(pred_logits, dim=-1)
                    
                    # Get class predictions (exclude background class)
                    scores, labels = torch.max(pred_probs[:, :-1], dim=-1)
                    
                    # Filter out low confidence predictions
                    keep = scores > 0.1
                    scores = scores[keep]
                    labels = labels[keep]
                    boxes = pred_boxes[keep]
                    
                    predictions.append({
                        'scores': scores.cpu(),
                        'labels': labels.cpu(),
                        'boxes': boxes.cpu()
                    })
                    
                    # Process targets
                    num_objects = batch['num_objects'][i].item()
                    if num_objects > 0:
                        target = {
                            'boxes': batch['boxes'][i][:num_objects].cpu(),
                            'labels': batch['labels'][i][:num_objects].cpu()
                        }
                    else:
                        target = {
                            'boxes': torch.empty(0, 4),
                            'labels': torch.empty(0, dtype=torch.long)
                        }
                    targets.append(target)
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, targets)
        
        return metrics, predictions, targets
    
    def visualize_predictions(self, predictions, targets, images, num_samples=5):
        """Visualize model predictions"""
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
        
        for i in range(min(num_samples, len(images))):
            # Original image with ground truth
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
            
            axes[0, i].imshow(img)
            axes[0, i].set_title('Ground Truth')
            axes[0, i].axis('off')
            
            # Draw ground truth boxes
            gt_boxes = targets[i]['boxes']
            for box in gt_boxes:
                x1, y1, x2, y2 = box * 512  # Scale to image size
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='green', linewidth=2)
                axes[0, i].add_patch(rect)
            
            # Image with predictions
            axes[1, i].imshow(img)
            axes[1, i].set_title('Predictions')
            axes[1, i].axis('off')
            
            # Draw prediction boxes
            pred_boxes = predictions[i]['boxes']
            pred_scores = predictions[i]['scores']
            
            for box, score in zip(pred_boxes, pred_scores):
                if score > 0.5:  # Only show confident predictions
                    x1, y1, x2, y2 = box * 512  # Scale to image size
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
                    axes[1, i].add_patch(rect)
                    axes[1, i].text(x1, y1-5, f'{score:.2f}', color='red', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_metrics(self, metrics):
        """Plot evaluation metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # mAP
        axes[0].bar(['mAP'], [metrics['mAP']])
        axes[0].set_title('Mean Average Precision')
        axes[0].set_ylim(0, 1)
        
        # Precision and Recall
        axes[1].bar(['Precision', 'Recall'], [metrics['precision'], metrics['recall']])
        axes[1].set_title('Precision and Recall')
        axes[1].set_ylim(0, 1)
        
        # AP per class
        class_names = [f"Class {i}" for i in range(len(metrics['AP_per_class']))]
        axes[2].bar(class_names, metrics['AP_per_class'])
        axes[2].set_title('Average Precision per Class')
        axes[2].set_ylim(0, 1)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('evaluation_metrics.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Example usage
    from src.rf_detr_model import build_rf_detr
    from src.data_loader import create_data_loaders
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path = "path/to/your/dataset"  # Update this
    
    # Load model
    model = build_rf_detr(num_classes=1).to(device)
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create data loader
    _, _, test_loader, categories = create_data_loaders(dataset_path, batch_size=4)
    
    # Evaluate
    evaluator = RFDETREvaluator(model, test_loader, device, categories)
    metrics, predictions, targets = evaluator.evaluate()
    
    print(f"mAP: {metrics['mAP']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Plot results
    evaluator.plot_metrics(metrics)
