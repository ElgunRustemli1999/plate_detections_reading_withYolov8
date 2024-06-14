#library
import numpy as np
import cv2
from ultralytics import YOLO
from reading import reading
import pytesseract




#Functions
def detect_plate(image,model_path,model_path_reading):
    print('[INFO].. Image is Loading..! ')
    image_array = np.asarray(image).astype(np.uint8)
    print('[INFO].. Process is Started..! ')
    model = YOLO(model_path)
    model_reading = YOLO(model_path_reading)
    number =''
    results = model(image_array)[0]

    is_detected = len(results.boxes.data.tolist())
   

    if is_detected is not 0:
        threshold = 0.5
        
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(score)
            cropped_images = image_array[y1:y2,x1:x2]
            if score > threshold:
                print()
                
                class_names = results.names[class_id]
                cv2.rectangle(image_array,(x1,y1),(x2,y2),(255,0,0),1)
                score = score*100
                #text = f'{class_names} : %{score:.2f}'
                text = pytesseract.image_to_string(cropped_images)
                #text = reading(cropped_images)
                print(text)
                
                cv2.putText(image_array,text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)

            # result_readings = model_reading(cropped_images)[0]


        return image_array,cropped_images, is_detected
    else:
        text = 'No Detections'
        cropped_images = np.zeros((512,512,3),np.uint8)
        cv2.putText(image_array,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)

        return image_array,cropped_images, is_detected


def detect_plate_video(image,model_path,model_path_reading):
    print('[INFO].. Image is Loading..! ')
    image_array = np.asarray(image).astype(np.uint8)
    print('[INFO].. Process is Started..! ')
    model = YOLO(model_path)
    model_reading = YOLO(model_path_reading)
    number =''
    results = model(image_array)[0]

    is_detected = len(results.boxes.data.tolist())
   

    if is_detected is not 0:
        threshold = 0.5
        
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(score)
            cropped_images = image_array[y1:y2,x1:x2]
            if score > threshold:
                print()
                
                class_names = results.names[class_id]
                cv2.rectangle(image_array,(x1,y1),(x2,y2),(255,0,0),1)
                score = score*100
                #text = f'{class_names} : %{score:.2f}'
                text = pytesseract.image_to_string(cropped_images)
                #text = reading(cropped_images)
                print(text)
                
                cv2.putText(image_array,text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)

            # result_readings = model_reading(cropped_images)[0]


        return image_array,cropped_images, is_detected
    else:
        text = 'No Detections'
        cropped_images = np.zeros((512,512,3),np.uint8)
        cv2.putText(image_array,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)

        return image_array,cropped_images, is_detected
