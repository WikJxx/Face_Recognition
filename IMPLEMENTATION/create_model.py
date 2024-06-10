import tensorflow as tf
from tensorflow import keras
from keras import layers

# Create the base model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
base_model.trainable = False  

# Add new layers on top of the base model
x = layers.GlobalAveragePooling2D()(base_model.output)  
x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), name="dense_128")(x)
x = layers.Dropout(0.3, name="dropout_1")(x)  
x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), name="dense_64")(x)
x = layers.Dropout(0.3, name="dropout_2")(x)  
final_output = layers.Dense(7, activation='softmax', name="output")(x)  

new_model = keras.Model(inputs=base_model.input, outputs=final_output)
new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

new_model.save("my_model.h5")
