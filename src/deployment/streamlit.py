import streamlit as st
from PIL import Image
from ultralytics import YOLO
import mlflow

import sys
sys.modules["torch.classes"] = None


# Instruction on how to use streamlit
# # To run this script, save it as `streamlit.py` and run the following command in your terminal:
# streamlit run streamlit.py
# Ensure you have the required packages installed:
# pip install streamlit ultralytics mlflow Pillow opencv-python




# Load trained model
#Local model loading ..


# use model from MLflow
@st.cache_resource

def load_model(local = False,path = None):

    if local :
        return YOLO(path) 
    
    else:
        mlflow.set_tracking_uri("https://dagshub.com/abdelrahman.samir/bone-fracture-detection.mlflow")
        client = mlflow.tracking.MlflowClient()

        model_versions = client.get_latest_versions(name="yolov8", stages=["None"])
        run_id = model_versions[0].run_id

        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="weights/best.pt")
        return YOLO(local_path)

model = load_model()


st.title("YOLOv8 Pone Fracture Detection")
st.markdown("Upload an image and run your trained model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Detection"):
        with st.spinner("Running detection..."):
            results = model.predict(image, save=False, augment=True)
            result_image = results[0].plot()
            st.image(result_image, caption="Detection Output", use_container_width=True)
