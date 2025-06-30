import os
import uuid
import random
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

#set paths
BASE_PATH = os.getcwd()
ANCH_PATH = os.path.join(BASE_PATH, 'img_data', 'anchors')
POS_PATH = os.path.join(BASE_PATH, 'img_data', 'positives')

def data_aug(img):
    data = []
    for i in range(5):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
        data.append(img)
    
    return data

for file_name in os.listdir(ANCH_PATH):
    img_path = os.path.join(ANCH_PATH, file_name)
    
    img = cv2.imread(img_path)
    if img is None:
        continue  # skip to next file
    augmented_images = data_aug(img) 
    # Ensure the image is resized to the expected dimensions  
    img = cv2.resize(img, (250, 250))  # Resize to 250x250 pixels
    
    for image in augmented_images:
        cv2.imwrite(os.path.join(ANCH_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())
for file_name in os.listdir(POS_PATH):
    img_path = os.path.join(POS_PATH, file_name)
    img = cv2.imread(img_path)
    if img is None:
        continue  # skip to next file
    img = cv2.resize(img, (250, 250))  # Ensure the image is resized to the expected dimensions
    augmented_images = data_aug(img) 
    
    for image in augmented_images:
        cv2.imwrite(os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())

