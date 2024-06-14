# Library
import cv2
from PIL import Image
import streamlit as st
from helper import detect_plate

# Title
st.title("Plate Reading System 🚗")

# Header
st.header("Upload an Image")

# Files
file = st.file_uploader("", type=["png", "jpg", "jpeg"])

# Model
model_path = "models/plate_reading.pt"

# Images
if file is not None:
    # Original Image
    st.header("Original Image")
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Processed Image
    st.header("Detection Result")
    detection_result = detect_plate(image, model_path)
    st.image(detection_result, use_column_width=True)

