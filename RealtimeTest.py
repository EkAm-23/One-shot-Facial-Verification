import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

BASE_PATH = os.getcwd()
INP_PATH = os.path.join(BASE_PATH,'app data' ,'input')
VERIF_PATH = os.path.join(BASE_PATH,'app data' ,'verification')

def preprocess(file_path):
    #reading the image from file path
    byte_img = tf.io.read_file(file_path)
    #loading the image
    img = tf.io.decode_jpeg(byte_img)
    #resizing image to 100x100
    img = tf.image.resize(img, (100,100))
    #scaling image to be between 0 and 1 
    img = img / 255.0
    return img

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    #calculating the difference between the anchor and the validation(positive or negative) images
    def call(self, inputStreams):
        input_embedding, validation_embedding = inputStreams
        return tf.math.abs(input_embedding - validation_embedding)
    

#verification function to compare the input image with the validation images
def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(VERIF_PATH):
        # Preprocess the images
        if not image.endswith('.jpg'):
            continue
        input_img = preprocess(os.path.join(INP_PATH, 'input_image.jpg'))
        validation_img = preprocess(os.path.join(VERIF_PATH, image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(VERIF_PATH))
    verified = verification > verification_threshold
    
    return results, verified


#LOADING THE PRETRAINED MODEL
siamese_model = tf.keras.models.load_model(os.path.join(BASE_PATH, 'siamese_model.keras'), custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

#taking input from the webcam and verifying the face
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    #frame is the image captured from the webcam
    frame = frame[:,400:1500, :] #cropping the frame to focus on the face
    frame = cv2.resize(frame, (250, 250)) #resizing the frame to 250x250 pixels(compressing the image)
    
    cv2.imshow('Verification', frame)
    
    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):

        cv2.imwrite(os.path.join(INP_PATH, 'input_image.jpg'), frame)
        # Run verification
        results, verified = verify(siamese_model, 0.5, 0.4)
        print(verified)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()