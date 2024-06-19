import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

Classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

new_model = tf.keras.models.load_model("MODULUL.h5")

test_directory = "IMPLEMENTATION\\test\\"
img_size = 224

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (img_size, img_size))
    image_normalized = image_resized / 255.0  # Normalize
    return image_normalized

total_images = 0
correct_predictions = 0

for category in Classes:
    path = os.path.join(test_directory, category)
    class_index = Classes.index(category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        
        preprocessed_image = preprocess_image(img_path)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
        
        predictions = new_model.predict(preprocessed_image)
        predicted_label_index = np.argmax(predictions)
        
        # Update total and correct predictions
        total_images += 1
        if predicted_label_index == class_index:
            correct_predictions += 1

accuracy = (correct_predictions / total_images) * 100

print(f"Total images: {total_images}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")
