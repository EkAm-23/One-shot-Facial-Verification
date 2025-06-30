#Import kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

#Import other dependencies
import cv2
import tensorflow as tf
import os
import numpy as np
from mtcnn import MTCNN

class CamApp(App):

    def build(self):
        # Main layout components 
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('embedding_model.keras')

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[:,400:1500, :] #cropping the frame to focus on the face

        # Flip horizontal and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and conver to 100x100px
    def preprocess(self, file_path):
        #reading the image from file path
        byte_img = tf.io.read_file(file_path)
        #loading the image
        img = tf.io.decode_jpeg(byte_img)
        #resizing image to 100x100
        img = tf.image.resize(img, (100,100))
        #scaling image to be between 0 and 1
        img = img / 255.0
        return img

    BASE_PATH = os.getcwd()
    detector = MTCNN()
    def crop_and_replace_with_face(self,img_path):
        # Read image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            # If the image cannot be read, return False
            return False

        # Convert BGR to RGB for MTCNN
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(img_rgb)

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

    # Verification function to verify person
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.4
        verification_threshold = 0.7

        # Capture input image from our webcam
        SAVE_PATH = os.path.join(self.BASE_PATH,'app data', 'input', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[:,400:1500, :] #cropping the frame to focus on the face
        cv2.imwrite(SAVE_PATH, frame)
        # Crop and replace the input image with the detected face
        if(self.crop_and_replace_with_face(SAVE_PATH)):
            Logger.info("Face cropped and saved: {}".format(SAVE_PATH))
        else:
            Logger.error("No face detected in the input image. Please try again.")
            self.verification_label.text = 'Verification Failed: No Face Detected'
            return
        
        # Preprocess the input image
        input_img = self.preprocess(SAVE_PATH)
        input_embedding = self.model.predict(tf.expand_dims(input_img, axis=0))[0]  # shape: (embedding_dim, )

        results = []
        for image in os.listdir(os.path.join(self.BASE_PATH,'app data', 'verification')):
            #if image is not a jpeg file, skip it
            if not image.endswith('.jpg'):
                continue
            # Preprocess the images
            validation_img = self.preprocess(os.path.join(self.BASE_PATH,'app data', 'verification', image))
            # Get the embedding for the validation image
            validation_embedding = self.model.predict(tf.expand_dims(validation_img, axis=0))[0]

            # Compute Euclidean distance (L2 distance)
            distance = np.linalg.norm(input_embedding - validation_embedding)
            results.append(distance)
        
        # Detection Threshold: Metric above which a prediciton is considered positive 
        detection = np.sum(np.array(results) < detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('app data', 'verification'))) 
        verified = verification > verification_threshold

        # Set verification text 
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

if __name__ == '__main__':
    CamApp().run()