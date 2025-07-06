# Real-Time Facial Verification System

This project implements a **real-time facial verification system** using a **Siamese Neural Network (SNN)** trained with **Triplet Loss**, and integrates it into a **Kivy-based desktop application**. The system compares the similarity between facial embeddings to verify identity with high accuracy.

## 🔍 Overview

- Trained a **Siamese Neural Network** with **CNN-based embedding generation** using **TensorFlow/Keras**.
- Utilized **Triplet Loss** for robust learning of similarity between facial features.
- Applied **MTCNN** for **face detection** and **extraction** from images and webcam input.
- Collected real-time images using **OpenCV**, applied **data augmentation**, and built a clean dataset.
- Built a **Kivy app** for real-time facial verification using the system webcam and saved model.

## Repository files include:

├── App # The directory which hold the final app code\
├── SNN_tripletLoss.ipynb # Contains the full code used to train the SNN model\
├── Positive_Anchor_Collect.py # Used to collect training data using webcam, and storing it into files\
├── dataAug.py # Used to expand the training dataset using data augmentation techniques\
├── extractFaces # Used to extract faces from training dataset using MTCNN face detector\
└── RealTimeTest.py # Code used for accessing the webcam for real time model testing\

### Inside the App directory

├── app data # Directory holding verification images for the app to use, and the input image captured\
├── facialVerif.py # The main app code used to run the verification model inside a kivy app UI\
├── model link # Contains a google drive link to the final trained model used for verification\
└── getVerif.py # Used to load more verification images in the app

## 🧠 Model Architecture

- **Base Network**: Convolutional Neural Network (CNN)
- **Loss Function**: Triplet Loss (with margin tuning)
- **Embedding Comparison**: L2 Distance used to compare anchor and validation(positive or negative) embeddings
- Implemented using **TensorFlow Functional API**

## 🖥️ Real-Time App

- Built with **Kivy** for cross-platform desktop use.
- Captures input from webcam, preprocesses and detects faces, compares embeddings in real-time.
- Verifies identity based on embedding similarity and a defined threshold.

## 🧪 Future Improvements

- Add multi-user enrollment and recognition support.
- Explore FaceNet or other pretrained backbone encoders
