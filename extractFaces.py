import cv2
import os
import numpy as np
from mtcnn import MTCNN

# Create directories
BASE_PATH = os.getcwd()
NEG_DATA = os.path.join(BASE_PATH, 'negative_images')
POS_DATA = os.path.join(BASE_PATH, 'positive_images')
ANCH_DATA = os.path.join(BASE_PATH, 'anchor_images')
data_list = [NEG_DATA, POS_DATA, ANCH_DATA]

NEG_PATH = os.path.join(BASE_PATH, 'negative_faces')
POS_PATH = os.path.join(BASE_PATH, 'positive_faces')
ANCH_PATH = os.path.join(BASE_PATH, 'anchor_faces')
destination_list = [NEG_PATH, POS_PATH, ANCH_PATH]
# Create directories if they do not exist
os.makedirs(NEG_PATH, exist_ok=True)
os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(ANCH_PATH, exist_ok=True)

# Initialize MTCNN detector
detector = MTCNN()

def extract_face(img_path, target_size=(100,100)):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    
    if results:
        x, y, w, h = results[0]['box']
        x, y = abs(x), abs(y)  # handle negative coordinates
        face = img_rgb[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        return face
    else:
        return None

# Extract faces from each data directory and save to corresponding destination directory
for i in range(len(data_list)):
    files = os.listdir(data_list[i])
    for file in files:
        source_path = os.path.join(data_list[i], file)
        if os.path.isfile(source_path):
            face = extract_face(source_path)
            if face is not None:
                destination_path = os.path.join(destination_list[i], file)
                cv2.imwrite(destination_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    print(f"Extracted faces from {data_list[i]} to {destination_list[i]}")