import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

# Load the trained model
new_model = tf.keras.models.load_model("trained_model.h5")

# Initialize the face cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Font parameters for text display
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
rectangle_bgr = (255, 255, 255)

# Initialize webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        if len(roi_color) > 0:
            face_roi = roi_color
            final_image = cv2.resize(face_roi, (224, 224))
            final_image = np.expand_dims(final_image, axis=0)  # 4 dimensions are needed
            final_image = final_image / 255.0  # Normalize

            Predictions = new_model.predict(final_image)
            predicted_label_index = np.argmax(Predictions)
            status = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"][predicted_label_index]

            # Draw the label on the frame
            x1, y1, w1, h1 = 0, 0, 175, 75
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), font, fontScale=font_scale, color=(0, 0, 255), thickness=2)
            cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)

    cv2.imshow("Face recognition", frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
