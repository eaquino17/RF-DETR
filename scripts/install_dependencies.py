import subprocess
import sys

def install_packages():
    """Install required packages for the RF-DETR project"""
    packages = [
        "torch",
        "torchvision", 
        "transformers",
        "pycocotools",
        "opencv-python",
        "matplotlib",
        "seaborn",
        "pillow",
        "numpy",
        "scipy",
        "tqdm",
        "roboflow"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

if __name__ == "__main__":
    install_packages()
