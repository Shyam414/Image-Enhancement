# Super-Resolution Convolutional Neural Network (SRCNN) Project

project Link: https://github.com/Shyam414/Image-Enhancement

# Overview


This project focuses on enhancing low-quality images using a Super-Resolution Convolutional Neural Network (SRCNN). The SRCNN model is trained to take low-resolution images as input and output high-resolution images. The dataset consists of pairs of low-quality and high-quality images.

# Project Structure
## normalize.py: 
Contains functions for image preprocessing such as loading, resizing, and normalizing images.
## plot.py: 
Contains functions for plotting original and enhanced images side by side.
## train.py: 
Main script for training the SRCNN model.
## srcnn_model.h5: 
Pre-trained SRCNN model.
## low_quality_images/: 
Directory containing low-quality images.
## high_quality_images/: 
Directory containing corresponding high-quality images.

## How to Use
Preprocess the Images: Ensure all low-quality and high-quality image pairs are processed and normalized.<br>
Train the Model: Run train.py to train the SRCNN model using the preprocessed image pairs.<br>
Test the Model: Use a low-quality test image, preprocess it, and use the trained SRCNN model to enhance it. Plot the original and enhanced images side by side.
## Requirements
Python 3.x<br>
TensorFlow<br>
NumPy<br>
Pillow<br>
Matplotlib

## Installation
Install the required packages using pip:<br>
pip install tensorflow numpy pillow matplotlib
## pip install tensorflow numpy pillow matplotlib
Ensure the low_quality_images and high_quality_images directories contain the appropriate image pairs.<br>
Execute the train.py script to train and save the model.<br>
Use the trained model to enhance new low-quality images and visualize the results using the plot_image function.

## Acknowledgements
This project uses the SRCNN architecture as described in the paper "Image Super-Resolution Using Deep Convolutional Networks" by Chao Dong et al.