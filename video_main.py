
import cv2
from PIL import Image
import streamlit as st 
from video import detect_plate_video
import numpy as np
import tempfile
# Main Streamlit application
st.title("Number Plate Detection in Video")

uploaded_video = st.file_uploader('Choose a video file', type=['mp4', 'mov', 'avi', 'mkv'])

# Model paths
model_path_detection = r'C:\Users\Gold\Desktop\yolov8_gpu\number_plate_reading_app\models\plate_detection.pt'
model_path_reading = r'C:\Users\Gold\Desktop\yolov8_gpu\number_plate_reading_app\models\plate_reading.pt'

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_video.read())
        video_path = temp_file.name

    st.header('Original Video')
    st.video(video_path)

    st.header("Detection Results")

    detect_result_path = detect_plate_video(video_path, model_path_detection, model_path_reading)

    if detect_result_path:
        st.header('Processed Video')
        st.video(r'C:\Users\Gold\Desktop\yolov8_gpu\number_plate_reading_app\output.avi')
    else:
        st.write("Error in processing video.")