# FaceEmotionRecognitionProject
TECHNOLOGIES: Python, Tensorflow, OpenCV

Goal of this project is to recognise emotions of a person in the real time.

FUNCTIONALITIES:
- 
- 
- 

How to use:
1. Adjust paths where your dataset is located
2. Run create_data.py
3. Run create_model,py
4. Run train_data.py (adjust epoch's number and if the accuracy is not enough for you run it again) 
5. You can check your model using multiple ways 
    * check_model.py - checks an image you uploaded
    * real_time_check.py - checks real time emotions using your camera 
    * test_data_check.py - checks your test data accuracy 

What do you need:
1. (OPTIONAL) Install MINICONDA (https://docs.anaconda.com/free/miniconda/)
2. Find ANACONDA PROMPT (or just terminal) and download TENSORFLOW (pip install tensorflow)
3. Download dataset (https://www.kaggle.com/datasets/msambare/fer2013)
4. Download OpenCV (pip install opencv-python)
5. Make sure you have numpy and matplotlib also installed.
6. Download cascade for frontal faces(face detection algorithm) https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml
7. Install deepface (pip install deepface) only for real time image