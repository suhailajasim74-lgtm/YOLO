import os
import subprocess
import sys

# Step 1: Install basic dependencies FIRST
print("Installing core packages...")
core_packages = [
    "numpy",
    "pillow",
    "pyyaml",
    "requests",
    "tqdm"
]

for package in core_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

# Step 2: Clone YOLOv5 if it doesn't exist
print("\nCloning YOLOv5 repository...")
if not os.path.exists('yolov5'):
    subprocess.check_call(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
else:
    print("YOLOv5 already cloned")

# Step 3: NOW install YOLOv5 dependencies
print("\nInstalling YOLOv5 dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "yolov5/requirements.txt", "-q"])

# Step 4: Install additional packages
print("\nInstalling additional packages...")
additional_packages = [
    "torch",
    "torchvision",
    "opencv-python",
    "matplotlib",
    "ipykernel",
    "jupyter"
]

for package in additional_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

print("\n✓ Setup complete!")