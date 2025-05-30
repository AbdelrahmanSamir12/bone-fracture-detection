import cv2
import os
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm  
from PIL import Image
from ultralytics import YOLO
import dvc.api

def process_dataset(cfg):
    paths = {"train": cfg["train_path"], "val": cfg["val_path"], "test": cfg["test_path"]}
    output_dir = cfg['output_dir']
    valid_ext = ['.png', '.jpg', '.jpeg']
    valid_classes = [str(i) for i in range(7)]
 
    for split, path in paths.items():
        img_dir = os.path.join(path, 'images')
        label_dir = os.path.join(path, 'labels')
        processed_img_dir = os.path.join(output_dir, split, 'images')
        processed_label_dir = os.path.join(output_dir, split, 'labels')
        os.makedirs(processed_img_dir, exist_ok=True)
        os.makedirs(processed_label_dir, exist_ok=True)
        for img_file in tqdm(os.listdir(img_dir), desc=f'Processing {split}'):
            img_path = os.path.join(img_dir, img_file)
            name, ext = os.path.splitext(img_file)
            label_file = f"{name}.txt"
            label_path = os.path.join(label_dir, label_file)
            
            invalid = False
            if ext.lower() not in valid_ext:
                invalid = True
            if not os.path.exists(label_path):
                invalid = True
            else:
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        first_char = content.split()[0]
                        if first_char not in valid_classes:
                            invalid = True
            if not invalid:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    resized_img = img.resize((640, 640))
                processed_img_path = os.path.join(processed_img_dir, img_file)
                resized_img.save(processed_img_path)
                shutil.copy2(label_path, os.path.join(processed_label_dir, label_file))
    print("Images Processed Successfully...")

if __name__ == "__main__":
    params = dvc.api.params_show("../params.yaml")
    process_dataset(params)