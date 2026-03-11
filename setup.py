import os
import subprocess
import sys

print("\n" + "="*60)
print("  YOLOv5 Sign Language Detection - Complete Setup")
print("="*60 + "\n")

# Step 1: Create data.yaml
print("Step 1: Creating data.yaml...")
data_yaml = """path: C:/Users/suhai/Desktop/YOLO
train: train/images
val: test/images

nc: 5
names: ['class1', 'class2', 'class3', 'class4', 'class5']
"""

with open('data.yaml', 'w') as f:
    f.write(data_yaml)
print("✓ data.yaml created\n")

# Step 2: Create custom model config
print("Step 2: Creating custom YOLOv5 config...")
yaml_content = """# Custom YOLOv5s parameters
nc: 5
depth_multiple: 0.33
width_multiple: 0.50

anchors:
  - [10,13, 16,30, 33,23]
  - [30,61, 62,45, 59,119]
  - [116,90, 156,198, 373,326]

backbone:
  [[-1, 1, Focus, [64, 3]],
   [-1, 1, Conv, [128, 3, 2]],
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, BottleneckCSP, [1024, False]],
  ]

head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],
   [-1, 3, BottleneckCSP, [512, False]],

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],
   [-1, 3, BottleneckCSP, [256, False]],

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],
   [-1, 3, BottleneckCSP, [512, False]],

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],
   [-1, 3, BottleneckCSP, [1024, False]],

   [[17, 20, 23], 1, Detect, [nc, anchors]],
  ]
"""

os.makedirs('yolov5/models', exist_ok=True)
with open('yolov5/models/custom_yolov5s.yaml', 'w') as f:
    f.write(yaml_content)
print("✓ Custom model config created\n")

# Step 3: Create hyperparameter files
print("Step 3: Creating hyperparameter files...")
hyp_content = """lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.20
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
"""

os.makedirs('yolov5/data/hyps', exist_ok=True)
with open('yolov5/data/hyps/hyp.scratch-low.yaml', 'w') as f:
    f.write(hyp_content)
with open('yolov5/data/hyps/hyp.scratch.yaml', 'w') as f:
    f.write(hyp_content)
print("✓ Hyperparameter files created\n")

# Step 4: Verify directory structure
print("Step 4: Verifying directory structure...")
required_dirs = ['train/images', 'train/labels', 'test/images', 'test/labels']
for dir_path in required_dirs:
    if os.path.exists(dir_path):
        file_count = len(os.listdir(dir_path))
        print(f"✓ {dir_path}/ ({file_count} files)")
    else:
        print(f"✗ {dir_path}/ NOT FOUND - Please create this directory!")

print("\n" + "="*60)
print("  Setup Complete! Ready to train.")
print("="*60)
print("\nNext step: Run training with:")
print("  cd yolov5")
print("  python train.py --img 416 --batch 16 --epochs 100 --data ../data.yaml --cfg ./models/custom_yolov5s.yaml --weights yolov5s.pt --name yolov5s_results\n")