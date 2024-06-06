import cv2
import os
import numpy as np
import random
import pickle

#Adjust data path
Datadirectory = "IMPLEMENTATION\\train\\"
Classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
img_size = 224

#Create function for training data
def create_training_Data():
    training_Data = []
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_Data.append([new_array, class_num])
            except Exception as e:
                pass
    return training_Data

training_Data = create_training_Data()

# Shuffle the training data
random.shuffle(training_Data)

# Split the data into four parts
chunk_size = len(training_Data) // 4
chunks = [training_Data[i*chunk_size : (i+1)*chunk_size] for i in range(4)]
if len(training_Data) % 4 != 0:
    chunks[-1].extend(training_Data[4*chunk_size:])

# Save each chunk as a separate file using pickle
for i, chunk in enumerate(chunks):
    with open(f'training_data_part_{i + 1}.pkl', 'wb') as f:
        pickle.dump(chunk, f)

print(f"Total training samples: {len(training_Data)}")
for i in range(4):
    print(f"Samples in part {i + 1}: {len(chunks[i])}")