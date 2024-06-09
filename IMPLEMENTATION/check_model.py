import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2

#Import trained model
new_model = tf.keras.models.load_model("trained_model.h5")

#Read the photo
frame = cv2.imread("disgust.webp")
print(frame.shape)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for x, y, w, h in faces:
    roi_gray = gray[y:y+h, x:x+w] #roi = region of interest :)\
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame,(x,y), (x+w, y+h),(255,0,0),2)
    facesdetect = faceCascade.detectMultiScale(roi_gray)
    if len(facesdetect)==0:
        print("Face not detected")
    else:
        for (ex,ey,ew,eh) in facesdetect:
            face_roi = roi_color[ey:ey+eh, ex:ex+ew]

final_image = cv2.resize(face_roi,(224,224))
final_image = np.expand_dims(final_image,axis = 0) #4 dinemsion is needed
final_image = final_image/255.0 # Normalizing

Predictions = new_model.predict(final_image)
Predictions[0]
print(np.argmax(Predictions))