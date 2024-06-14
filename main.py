#Library
import cv2
from PIL import Image
import streamlit as st 
from helper import detect_plate,detect_plate_video
import numpy as np
import tempfile




#titles

st.title('Welcome To Car Plate Recognition Systems 🚗')


#haeder
st.header('Upload an Image')

#files
file = st.file_uploader('Choose an image file',type=['jpeg','png','jpg'])
#file_video = st.file_uploader('Choose a video file', type=['mp4', 'mov', 'avi', 'mkv'])



#model
model_path_detection = r'C:\Users\Gold\Desktop\yolov8_gpu\number_plate_reading_app\models\plate_detection.pt'
model_path_reading = r'C:\Users\Gold\Desktop\yolov8_gpu\number_plate_reading_app\models\plate_reading.pt'

#image
if file is not None:
    #orginal image
    img =Image.open(file).convert('RGB')
    st.header('Orginal image')
    st.image(image=img,use_column_width=True)
    #processed image
    st.header("Detection Results")
    ### Func
    detect_result, cropped_image, is_detected = detect_plate(img,model_path=model_path_detection,model_path_reading=model_path_reading)
    if is_detected is not 0:
        st.write('#### [INFO]... Plate is detected !')

        st.image(detect_result,use_column_width=True)
        st.image(cropped_image, use_column_width=True)
    else:
        st.write('#### [INFO].... Plate is not detected')
        st.image(detect_result,use_column_width=True)
#video
# Video file uploader





