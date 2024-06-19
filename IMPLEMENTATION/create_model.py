import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

#Import pretrained model to be a base of our new model
base_model = tf.keras.applications.MobileNetV2()
base_output = base_model.layers[-2].output

final_output = layers.Dense(128)(base_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output= layers.Dense(7,activation='softmax')(final_output) #clasification layer

new_model = keras.Model(inputs = base_model.input, outputs = final_output)
new_model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
#Save the model
new_model.save("MODEL123.h5")