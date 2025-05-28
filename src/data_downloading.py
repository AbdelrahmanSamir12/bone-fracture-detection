import os
import zipfile
import dvc.api
from kaggle.api.kaggle_api_extended import KaggleApi


def download_data(cfg):
    # Set Kaggle config path - ensure this directory exists and contains kaggle.json
    os.environ['KAGGLE_CONFIG_DIR'] = '/teamspace/studios/this_studio/bone-fracture-detection/.config/kaggle'
    # Set your specific data path
    raw_data_path = cfg['raw_data_path']
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()  
    
    # Download dataset
    dataset_name = cfg["dataset_name"]
    api.dataset_download_files(dataset_name, path=raw_data_path, unzip=True)
    
    print(f"Data downloaded and extracted to {raw_data_path}")

if __name__ == "__main__":
    params = dvc.api.params_show("../params.yaml")
    download_data(params)