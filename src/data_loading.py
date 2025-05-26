import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_data():
    # Set your specific data path
    raw_data_path = "data/raw"
    
    # Ensure data directory exists
    #os.makedirs(raw_data_path, exist_ok=True)
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Dowload Telecom Customer Churn dataset
    dataset_name = "pkdarabi/bone-fracture-detection-computer-vision-project"
    #zip_file_path = os.path.join(raw_data_path, "telecom-customer-churn-dataset.zip")
    api.dataset_download_files(dataset_name, path=raw_data_path, unzip=True)
    
    
    
    
   
    
    print(f"Data downloaded and extracted to {raw_data_path}")

if __name__ == "__main__":
    download_data()