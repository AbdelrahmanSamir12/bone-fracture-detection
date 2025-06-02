import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO
import torch
import os

@hydra.main(version_base=None, config_path="..", config_name="params")
def main(cfg: DictConfig) -> None:
    print("Loaded configuration:")
    print(cfg)
    # os.environ["MLFLOW_TRACKING_USERNAME"] = "your_username"
    # os.environ["MLFLOW_TRACKING_PASSWORD"] = "your_token"
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    mlflow.autolog()
    
    # List existing experiments
    client = MlflowClient()
    print("\nExisting experiments:")
    for exp in client.search_experiments():
        print(f"  - {exp.name} (ID: {exp.experiment_id})")
    
    # Initialize and train model
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {'GPU' if device == 0 else 'CPU'}")
    
    model = YOLO(cfg.train.model)
    model.train(
        data=cfg.train.data,
        augment=cfg.train.augment,
        device=device,
        imgsz=cfg.train.imgsz,
        epochs=cfg.train.epochs,
        batch=cfg.train.batch,
        verbose=True,
        name=cfg.train.name,
    )
    
    # Save model in Hydra's output directory
    output_path = os.path.join(os.getcwd(), "yolov8s.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nSaved model to: {output_path}")
    mlflow.log_artifact(output_path)  # Log to MLflow

if __name__ == "__main__":
    main()