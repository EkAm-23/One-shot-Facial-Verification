import cv2
import os
import matplotlib.pyplot as plt
import uuid # to generate unique filenames for images


#define paths to anchor and positive images
BASE_PATH = os.getcwd() #path to the current working directory
ANCHOR_PATH = os.path.join(BASE_PATH, 'img_data', 'anchors') #path to anchor images
POSITIVE_PATH = os.path.join(BASE_PATH, 'img_data', 'positives') #path to positive images

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

    #COLLECTING ANCHOR IMAGES
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Create the unique file path 
        imgname = os.path.join(ANCHOR_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out anchor image
        cv2.imwrite(imgname, frame)
        
    #COLLECTING POSITIVE IMAGES
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # Create the unique file path 
        imgname = os.path.join(POSITIVE_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out positive image
        cv2.imwrite(imgname, frame)
        

    #DISPLAY the frame in a window named 'Frame'
    cv2.imshow('Frame', frame)
    
    #to BREAK the loop and stop capturing frames, press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


