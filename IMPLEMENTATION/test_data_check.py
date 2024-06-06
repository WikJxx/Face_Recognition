import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Define the class labels
Classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Load the trained model
new_model = tf.keras.models.load_model("final_model.h5")

# Path to the test images directory
test_directory = "IMPLEMENTATION\\test\\"
img_size = 224

# Function to preprocess images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (img_size, img_size))
    image_normalized = image_resized / 255.0  # Normalize
    image_expanded = np.expand_dims(image_normalized, axis=0)  # Add batch dimension
    return image_expanded

# Function to display images with predicted labels
def display_image_with_prediction(image, prediction):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Predicted: {prediction}')
    plt.axis('off')
    plt.show()

# Iterate through the test images
for category in Classes:
    path = os.path.join(test_directory, category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)
        
        # Preprocess the image
        preprocessed_image = preprocess_image(img_path)
        
        # Make prediction
        predictions = new_model.predict(preprocessed_image)
        predicted_label_index = np.argmax(predictions)
        predicted_label = Classes[predicted_label_index]
        
        # Display the image with the predicted label
        display_image_with_prediction(image, predicted_label)