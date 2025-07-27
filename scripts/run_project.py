"""
Complete RF-DETR Object Detection Project Runner
This script runs the entire project pipeline from data download to evaluation.
"""

import os
import sys
import subprocess

def run_step(step_name, script_path, description):
    """Run a project step and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"Description: {description}")
    print('='*60)
    
    try:
        if script_path.endswith('.py'):
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True, check=True)
            print(result.stdout)
            if result.stderr:
                print("Warnings/Errors:", result.stderr)
        else:
            # For non-Python commands
            result = subprocess.run(script_path.split(), 
                                  capture_output=True, text=True, check=True)
            print(result.stdout)
        
        print(f"✅ {step_name} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} failed!")
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ {step_name} failed with unexpected error: {e}")
        return False

def main():
    """Run the complete RF-DETR project pipeline"""
    
    print("🚀 Starting RF-DETR Object Detection Project")
    print("This will run the complete pipeline:")
    print("1. Install dependencies")
    print("2. Download dataset")
    print("3. Train RF-DETR model")
    print("4. Evaluate model performance")
    print("5. Generate visualizations")
    
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Install dependencies
    success = run_step(
        "Install Dependencies",
        "scripts/install_dependencies.py",
        "Installing required Python packages for the project"
    )
    if not success:
        print("⚠️  Dependency installation failed. You may need to install packages manually.")
    
    # Step 2: Download dataset
    success = run_step(
        "Download Dataset",
        "scripts/download_dataset.py",
        "Downloading car detection dataset from Roboflow"
    )
    if not success:
        print("❌ Dataset download failed. Please check your internet connection and API key.")
        return
    
    # Step 3: Run main training and evaluation
    success = run_step(
        "Train and Evaluate Model",
        "scripts/main_training.py",
        "Training RF-DETR model and evaluating performance"
    )
    if not success:
        print("❌ Training/evaluation failed. Check the error messages above.")
        return
    
    print("\n" + "🎉" * 20)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("🎉" * 20)
    
    print("\n📊 Results Summary:")
    print("- Model checkpoints saved in 'checkpoints/' directory")
    print("- Training history saved as JSON")
    print("- Evaluation metrics plotted and saved")
    print("- Prediction visualizations generated")
    
    print("\n📁 Generated Files:")
    print("- checkpoints/best_model.pth (best model weights)")
    print("- checkpoints/training_history.json (training curves)")
    print("- predictions_visualization.png (sample predictions)")
    print("- evaluation_metrics.png (performance metrics)")
    
    print("\n🔍 Key Metrics to Review:")
    print("- Mean Average Precision (mAP)")
    print("- Precision and Recall")
    print("- Training/Validation loss curves")
    print("- Visual inspection of predictions")
    
    print("\n📝 Next Steps:")
    print("1. Review the generated visualizations")
    print("2. Analyze the training curves for overfitting")
    print("3. Experiment with hyperparameters if needed")
    print("4. Try data augmentation techniques")
    print("5. Consider ensemble methods for better performance")

if __name__ == "__main__":
    main()
