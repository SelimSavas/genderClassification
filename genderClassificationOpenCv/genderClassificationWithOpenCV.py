import tensorflow

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
               
ths = open("log.txt", "w")


model = tensorflow.keras.models.load_model('model.h5')

webcam = cv2.VideoCapture(0)

classes = ['female','male']

while webcam.isOpened():

    
    status, frame = webcam.read()

    face, confidence = cv.detect_face(frame)

    for idx, f in enumerate(face):
      
        (startX, startY) = f[0]-60, f[1]-60
        (endX, endY) = f[2]+60, f[3]+60
        
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        face_crop = np.copy(frame[startY:endY,startX:endX])
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue
        
        face_crop = cv2.resize(face_crop, (178,218))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        
        conf = model.predict(face_crop)[0]
        
        idx = np.argmax(conf)
        label = classes[idx]
        
        if (conf[idx]>0.90):
            label = "{}: {:.2f}%".format(label, conf[idx]*100)
            cv2.putText(frame, label, (startX, startY),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
       
    ths.write("Erkek geldi\n")


    cv2.imshow("gender detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
ths.close()
webcam.release()
cv2.destroyAllWindows()