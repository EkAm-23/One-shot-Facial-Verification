import cv2
import os
import matplotlib.pyplot as plt
import uuid # to generate unique filenames for images
from mtcnn import MTCNN

detector = MTCNN() # Initialize the MTCNN face detector

BASE_PATH = os.getcwd() #path to the current working directory
VERIF_PATH = os.path.join(BASE_PATH, 'app data', 'verification') #path to verification images
#empty the verification folder if it exists
if os.path.exists(VERIF_PATH):
    for file in os.listdir(VERIF_PATH):
        file_path = os.path.join(VERIF_PATH, file)
        os.remove(file_path) 



cap= cv2.VideoCapture(0) #establishing a connection to the webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
#loop to continuously capture frames from the webcam 

while cap.isOpened():
    ret, frame = cap.read()

    #frame is the image captured from the webcam
    frame = frame[:,400:1500, :] #cropping the frame to focus on the face
    frame = cv2.resize(frame, (250, 250)) #resizing the frame to 250x250 pixels(compressing the image)

    #COLLECTING IMAGES
    if cv2.waitKey(1) & 0XFF == ord('v'):
        # Create the unique file path 
        imgname = os.path.join(VERIF_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out image
        cv2.imwrite(imgname, frame)
        

    #DISPLAY the frame in a window named 'Frame'
    cv2.imshow('Frame', frame)

    #to BREAK the loop and stop capturing frames when 50 images are collected
    if len(os.listdir(VERIF_PATH)) >= 50:
        break

cap.release()
cv2.destroyAllWindows()

def crop_and_replace_with_face(img_path):
    # Read image using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        # If the image cannot be read, return False
        return False
    # Convert BGR to RGB for MTCNN
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    if results:
        x, y, width, height = results[0]['box']
        x, y = max(0, x), max(0, y)
        face = img[y:y+height, x:x+width]

        # Overwrite original image with cropped face
        cv2.imwrite(img_path, face)
        return True
    else:
        # If no face is detected, return the original image
        return False

for image in os.listdir(VERIF_PATH):
    #if image is not a jpeg file, skip it
    if not image.endswith('.jpg'):
        continue
    img_path = os.path.join(VERIF_PATH, image)
    crop_and_replace_with_face(img_path)
