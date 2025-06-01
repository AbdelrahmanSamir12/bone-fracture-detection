import cv2
import os
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm  
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict
import random
from ultralytics import YOLO
import mlflow

from mlflow.tracking import MlflowClient

from pytorch_lightning import Trainer

trainer = Trainer(accelerator="cpu", devices="auto")
# os.environ["MLFLOW_TRACKING_USERNAME"] = "your_username"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "your_token"


mlflow.set_tracking_uri("https://dagshub.com/abdelrahman.samir/bone-fracture-detection.mlflow")

experiment_name = "bonefracture_yolov8s"
# mlflow.create_experiment(experiment_name)
mlflow.autolog()
client = MlflowClient()

experiments = client.search_experiments()
for exp in experiments:
    print(f"ID: {exp.experiment_id}, Name: {exp.name}")

mlflow.set_experiment(experiment_name)

model = YOLO('yolov8s.pt')
# model.set_class_weights(class_weights)
model.train(
    data="/teamspace/studios/this_studio/bone-fracture-detection/data/data.yaml",
    augment=True, 
    device = 'cpu',
    imgsz=640,
    epochs=50,
    batch=16,
    verbose=True,
    name="bonefracture_yolov8s",
)
