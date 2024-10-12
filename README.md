Project Overview
 
This repository contains two Python scripts related to a Convolutional Neural Network (CNN) model designed to recognize whether an object is a fork or a knife, as well as a script for real-time object recognition using a camera.


1. CNNforkorknife.py

The CNNforkorknife.py script implements a CNN model to classify images as either a fork or a knife. The dataset I used for training and validation was sourced from Kaggle. Before training the model, I resized and categorized the images to fit the requirements of the neural network. The main goal of this project was to strengthen my knowledge of artificial intelligence, specifically in the area of deep learning, by working with a simple image recognition task.
The CNN is structured with layers such as convolutional layers, pooling layers, and fully connected layers. The network is trained on labeled images of forks and knives, learning to distinguish between the two based on visual features.


2. camfork.py
   
The camfork.py script is designed to test the CNN model in real-time by using the camera on my device. This script allows the model to recognize whether the object in my hand is a fork or a knife. The goal was to experiment with the camera input and see how well the model performs on real-world objects.
This script also serves as a foundation for future projects that will involve real-time object recognition using a camera. My aim is to develop more complex projects where the camera can identify various items or features in real-time, expanding on the skills learned in this basic fork/knife classification project.
