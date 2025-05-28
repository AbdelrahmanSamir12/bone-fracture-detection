from src.data_downloading import download_data
from src.data_processing import process_dataset
import dvc.api
import os

def main():
    params = dvc.api.params_show("params.yaml")

    raw_data_exists = os.path.exists(params['train_path']) and \
                     os.path.exists(params['val_path']) and \
                     os.path.exists(params['test_path'])
    
    if not raw_data_exists:
        print("Raw data not found. Downloading data...")
        download_data(params)
    else:
        print("Raw data already exists. Skipping download.")
    
    processed_data_exists = os.path.exists(params['output_dir'])
    
    if not processed_data_exists:
        print("Processed data not found. Processing data...")
        process_dataset(params)
    else:
        print("Processed data already exists. Skipping processing.")

if __name__ == "__main__":
    main()