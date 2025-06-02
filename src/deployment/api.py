import numpy as np
import litserve as ls
import pickle
import pandas as pd
import mlflow
from ultralytics import YOLO
import os
from PIL import Image


class InferenceAPI(ls.LitAPI):
    def setup(self, device="cpu"):
        mlflow.set_tracking_uri("https://dagshub.com/abdelrahman.samir/bone-fracture-detection.mlflow")
        client = mlflow.tracking.MlflowClient()
        
        model_versions = client.get_latest_versions(name="yolov8", stages=["None"])
        run_id = model_versions[0].run_id
        
        local_dir = f"models/yolov8/{run_id}"
        local_path = os.path.join(local_dir, "best.pt")

        if not os.path.exists(local_path):
            print("Model not found locally. Downloading from MLflow...")
            os.makedirs(local_dir, exist_ok=True)
            downloaded_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="weights/best.pt")
            os.rename(downloaded_path, local_path)
        else:
            print("Model found locally. Skipping download.")

        self.model = YOLO(local_path)


    def predict(self, x):
        if x is not None:
            image = np.array(x['image'])
            results = self.model.predict(image, conf=0.5)
            return results
        else:
            return None

    def encode_response(self, output):
        if output is None:
            return {
                "message": "Error Occurred",
                "prediction": None
            }
        
        predictions = []
        for result in output:
            detection = {
                "boxes": [],
                "scores": [],
                "classes": [],
                "class_names": []
            }
            
            if result.boxes is not None and len(result.boxes):
                detection["boxes"] = result.boxes.xyxy.cpu().numpy().tolist()
                detection["scores"] = result.boxes.conf.cpu().numpy().tolist()
                detection["classes"] = result.boxes.cls.cpu().numpy().tolist()
                detection["class_names"] = [result.names[int(cls)] for cls in detection["classes"]]
            
            predictions.append(detection)
        
        return {
            "message": "Response Produced Successfully",
            "prediction": predictions,
            "original_shape": output[0].orig_shape if output else None,
            "model_input_shape": (640, 640)
        }


