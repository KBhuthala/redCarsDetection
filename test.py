# -*- coding: utf-8 -*-
"""
File:   test.py
Author: Keerthana Bhuthala
Date: 10/07/2018
Desc: Solution for Project00

"""
import numpy as np

def testing(X,Y,knn_,testing_data):
    data_test = testing_data
    
    #Creating lists for storing RGB values of Test data and Labels for the Test data
    Test_RGB = [] 
            
    #Getting all the RGB values of the pixels in the RGB list but to test, only a subset is considered where
    # i is in range (50,170) and j is in range (1517,1985)
    for i in range(0,6250):
        for j in range(0,6250):
            Test_RGB.append(data_test[i,j])
                    
    #print (len(Test_RGB))
    
    # Retrieving the input parameter, training data set and the corresponding labels                
    X_train = X
    X_test = Test_RGB  
    y_train = Y 
    y_test = []
    
    #KNN Classifier                
    knn = knn_
    knn.fit(X_train, y_train)
    y_test = knn.predict(X_test)
    

#Creating a new 2D array that stores the values of the class labels 
    label_data_test = np.zeros((6250,6250))
    r=0
    c=0
    for i in range(0,len(y_test)):
        if((i%6250 == 0) and (i!=0)):
            r=r+1
            c=0
        label_data_test[r,c] = y_test[i]
        c=c+1

#Now, the array of labels is checked if is equal to 1, then corresponding X,Y coordinates are returned   
    for i in range(0,6250):
        for j in range(0,6250):
            if(label_data_test[i,j] == 1):
                print (i,j)
        
            
            




   
        