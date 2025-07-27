import torch
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.trainer import RFDETRTrainer
from src.evaluator import RFDETREvaluator
from src.rf_detr_model import build_rf_detr
from src.data_loader import create_data_loaders

def main():
    """Main training and evaluation pipeline"""
    
    # Configuration
    DATASET_PATH = "Car-Detection-1"  # Update this to your dataset path
    NUM_CLASSES = 1  # Car detection (single class)
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    print(f"Dataset path: {DATASET_PATH}")
    
    # Step 1: Train the model
    print("\n" + "="*50)
    print("STEP 1: TRAINING RF-DETR MODEL")
    print("="*50)
    
    trainer = RFDETRTrainer(
        dataset_path=DATASET_PATH,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )
    
    # Train the model
    trainer.train(num_epochs=NUM_EPOCHS)
    
    # Step 2: Evaluate the model
    print("\n" + "="*50)
    print("STEP 2: EVALUATING RF-DETR MODEL")
    print("="*50)
    
    # Load best model
    model = build_rf_detr(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test data loader
    _, _, test_loader, categories = create_data_loaders(DATASET_PATH, batch_size=BATCH_SIZE)
    
    # Evaluate
    evaluator = RFDETREvaluator(model, test_loader, DEVICE, categories)
    metrics, predictions, targets = evaluator.evaluate()
    
    # Print results
    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"Mean Average Precision (mAP): {metrics['mAP']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    for i, ap in enumerate(metrics['AP_per_class']):
        print(f"AP for class {i}: {ap:.4f}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # Get some test images for visualization
    test_images = []
    test_targets = []
    test_predictions = []
    
    for i, batch in enumerate(test_loader):
        if i >= 1:  # Only get first batch for visualization
            break
        test_images = batch['image']
        for j in range(len(batch['image'])):
            num_objects = batch['num_objects'][j].item()
            if num_objects > 0:
                target = {
                    'boxes': batch['boxes'][j][:num_objects],
                    'labels': batch['labels'][j][:num_objects]
                }
            else:
                target = {
                    'boxes': torch.empty(0, 4),
                    'labels': torch.empty(0, dtype=torch.long)
                }
            test_targets.append(target)
        
        # Get corresponding predictions
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + len(batch['image'])
        test_predictions = predictions[start_idx:end_idx]
    
    # Create visualizations
    evaluator.visualize_predictions(test_predictions, test_targets, test_images)
    evaluator.plot_metrics(metrics)
    
    print("\nTraining and evaluation completed!")
    print("Check the generated visualization files:")
    print("- predictions_visualization.png")
    print("- evaluation_metrics.png")
    print("- Model checkpoints in 'checkpoints/' directory")

if __name__ == "__main__":
    main()
