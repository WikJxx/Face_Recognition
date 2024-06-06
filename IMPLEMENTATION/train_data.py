import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import random
import pickle

# List of the split training data files
data_files = ['training_data_part_2.pkl','training_data_part_4.pkl','training_data_part_1.pkl','training_data_part_3.pkl']
img_size = 224

# Load the model
new_model = tf.keras.models.load_model("final_model.h5")
new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

for data_file in data_files:
    with open(data_file, 'rb') as f:
        training_Data = pickle.load(f)
    
    # Shuffle the training data
    random.shuffle(training_Data)

    X = []
    y = []
    for features, label in training_Data:
        X.append(features)
        y.append(label)

    # Convert lists to numpy arrays
    X = np.array(X).reshape(-1, img_size, img_size, 3)
    X = X / 255.0  # Normalize
    Y = np.array(y)

    print(f"Training on {data_file}:")
    print(X.shape)
    print(y[0])
    
    # Train the model
    new_model.fit(X, Y, epochs=10)
# Save the trained model
new_model.save("model.h5")
