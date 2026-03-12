# test_yaml.py
import yaml
from pathlib import Path

yaml_path = Path("C:/Users/suhai/Desktop/YOLO/data.yaml")
print(f"Testing YAML file: {yaml_path}")

if yaml_path.exists():
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        print("\nYAML contents:")
        print(data)
        
        if 'names' in data:
            names = data['names']
            print(f"\nClasses found: {names}")
            if isinstance(names, dict):
                print("Dictionary format:")
                for k, v in names.items():
                    print(f"  {k}: {v}")
            elif isinstance(names, list):
                print("List format:")
                for i, name in enumerate(names):
                    print(f"  {i}: {name}")
        else:
            print("No 'names' field found!")
else:
    print("YAML file not found!")