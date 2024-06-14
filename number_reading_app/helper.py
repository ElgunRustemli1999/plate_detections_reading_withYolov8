# Library
import cv2
import numpy as np
from ultralytics import YOLO

# Function
def detect_plate(image, model_path):

    green_color = (0,255,0)
    blue_color = (0,0,255) # BGR -- RGB
    font = cv2.FONT_HERSHEY_SIMPLEX

    print("[INFO].. Image is loading !")
    image_array = np.asarray(image)

    print("[INFO].. Processing is started !")
    model = YOLO(model_path)
    results = model(image_array)[0]

    is_detected = len(results.boxes.data.tolist())

    if is_detected is not 0:
        threshold = 0.5
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if score > threshold:
                cv2.rectangle(image_array, (x1,y1), (x2,y2), green_color, 2)
                
                score = score * 100
                class_name = results.names[class_id]
                
                # text = f"{class_name}: %{score:.2f}"
                cv2.putText(image_array, class_name, (x1,y1-10), font, 1, green_color, 2, cv2.LINE_AA)

    else:
        text = "No Detection"
        cv2.putText(image_array, text, (10,30), font, 1, blue_color, 2, cv2.LINE_AA)

    return image_array








    return image

