# Import necessary libraries
import numpy as np
import cv2
from ultralytics import YOLO
import pytesseract
import streamlit as st
import tempfile

def detect_plate_video(video_path, model_path, model_path_reading):
    print('[INFO].. Image is Loading..! ')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get the frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_path = 'output.avi'
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (frame_width, frame_height))

    # Load the models once
    model = YOLO(model_path)
    model_reading = YOLO(model_path_reading)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_array = np.asarray(frame).astype(np.uint8)
        print('[INFO].. Process is Started..! ')

        results = model(image_array)[0]
        is_detected = len(results.boxes.data.tolist())

        if is_detected != 0:
            threshold = 0.2
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if score > threshold:
                    cropped_images = image_array[y1:y2, x1:x2]
                    text = pytesseract.image_to_string(cropped_images)
                    print(text)
                    cv2.rectangle(image_array, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.putText(image_array, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        else:
            text = 'No Detections'
            cv2.putText(image_array, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        out.write(image_array)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return out_path