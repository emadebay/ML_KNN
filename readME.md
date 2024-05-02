This is the first Homework for the statistical machine learning course taught at Purdue University, Indianapolis fall 2023

The Homework requirement is in the file Homework_1-3.pdf and the written solution is in the solution.pdf

The programming implementation is in the KNN.py

KNN Implementation Readme
This repository contains a Python implementation of the K-Nearest Neighbors (KNN) algorithm for classification tasks. The implementation includes functions to read input data, compute the Hamming distance between data samples, and predict the class labels using the KNN algorithm.

Dataset Preparation
The dataset used for training the KNN model is provided in the data.txt file. Each feature value in the dataset is converted to binary (0 or 1), where 'skips' is represented as 1 and 'reads' as 0.

Functionality Overview
Read Data Function (read_data): This function reads the input data from the data.txt file and prepares it for further processing.
Distance Calculation Function (distance): The distance function computes the Hamming distance between two data samples, represented as vectors.
KNN Prediction Function (knn_predict): This function implements the KNN algorithm for predicting the class labels of test samples based on their nearest neighbors in the training data.
Testing
To test the KNN implementation, the provided test sample is used. The test sample contains the following attributes: Action, Author, Thread Length, and Where. The knn_predict function is called with different values of k (k=1 and k=3), and the predicted class labels are obtained.
