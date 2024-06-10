import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split

# List of the split training data files
data_files = ['training_data_part_1.pkl']
img_size = 224

# Load the model
new_model = tf.keras.models.load_model("second_training_model.h5")

# Add L2 regularization to the existing layers
def add_regularization(model, l2_factor=0.001):
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            layer.kernel_regularizer = keras.regularizers.l2(l2_factor)
    return model

new_model = add_regularization(new_model)

new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Define the data augmentation parameters
data_augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

for data_file in data_files:
    with open(data_file, 'rb') as f:
        training_data = pickle.load(f)
    
    # Shuffle the training data
    random.shuffle(training_data)

    X = []
    y = []
    for features, label in training_data:
        X.append(features)
        y.append(label)

    # Convert lists to numpy arrays
    X = np.array(X).reshape(-1, img_size, img_size, 3)
    X = X / 255.0  # Normalize
    Y = np.array(y)

    print(f"Training on {data_file}:")
    print(X.shape)
    print(y[0])
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)

    # Train the model with early stopping and data augmentation
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    new_model.fit(data_augmentation.flow(X_train, y_train),
                  epochs=25,
                  validation_data=(X_val, y_val),
                  callbacks=callbacks,
                  shuffle=True)

# Save the trained model
new_model.save("third_training_model.h5")
