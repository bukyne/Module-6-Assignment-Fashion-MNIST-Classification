# Module-6-Assignment-Fashion-MNIST-Classification
This project implements a Convolutional Neural Network (CNN) in Python and R using Keras to classify images from the Fashion MNIST dataset. The solution includes data loading, preprocessing, model building, training, evaluation, and prediction on sample images.

Dataset Details
The Fashion MNIST dataset includes grayscale images of size 28x28 pixels. The dataset is divided into 10 classes:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

Requirements
Python 3.11
R 4.5
Tensorflow 2.19
Python libraries: numpy, keras, tensorflow, pandas, matplotlib
R libraries: keras, tensorflow, ggplot2, gridExtra

Files
Fashion MNIST Classification.ipynb 
R Code for Fashion MNIST Classification.R 
README  

Solution Approach
To solve this assignment, we'll implement a Convolutional Neural Network (CNN) with exactly six layers to classify the Fashion MNIST dataset. The solution includes both Python and R implementations. The approach involves:

1. Data Loading and Preprocessing: Load the Fashion MNIST dataset and preprocess images by normalizing pixel values and reshaping for CNN input.
2. Model Architecture: Construct a 6-layer CNN with:
	Two convolutional layers for feature extraction
	Pooling layers for dimensionality reduction
	Flatten layer to transition to dense layers	  
 	Dense layers for classification
3. Model Training: Compile and train the model using appropriate loss functions and optimizers.
4. Prediction: Make predictions on test images and visualize results.

Setup Instructions

1. For Python - Install the required package: 
Install matplotlib numpy and keras by using "pip install matplotlib numpy keras" in the command prompt. Installing tensorflow.

Tip: Install tensorflow using this guide on youtube (https://www.youtube.com/watch?v=0w-D6YaNxk8).
2. For R - Install the required package:
Install keras, tensorflow and ggplot2
3. Run the Python Implementation script
4. Run the R implementation script

Results:
 - Trains a CNN model for 10 epochs
 - Prints test accuracy
	Python: ~88% test accuracy
	R: ~88% test accuracy
 - Predictions for two sample images are displayed with their predicted class labels.
 - Displays predictions for two test images

Model Architecture
The 6-layer CNN consists of:
 -Conv2D (32 filters, 3x3 kernel, ReLU)
 -MaxPooling2D (2x2 pool size)
 -Conv2D (64 filters, 3x3 kernel, ReLU)
 -MaxPooling2D (2x2 pool size)
 -Flatten
 -Dense (10 units, softmax)
