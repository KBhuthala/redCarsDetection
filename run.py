# -*- coding: utf-8 -*-
"""
File:   run.py
Author: Keerthana Bhuthala
Date: 10/07/2018
Desc: Solution for Project00

"""
import train
import test
import numpy as np

#Creating two lists to store the training data and target labels from train.py
X =[]
Y =[]

training_data = np.load('data_train.npy', 'r')
testing_data = np.load('data_test.npy','r')

# calling the  function in train.py to train the data and perform cross validation
X,Y,knn_ = train.training(training_data)

# The X and Y lists that are retrieved from train.py and given as inputs to the test,py
# calling the function in test.py to test the data and predict class labels for the test data
test.testing(X,Y,knn_,testing_data)
