"""
File:   train.py
Author: Keerthana Bhuthala
Date: 10/07/2018
Desc: Solution for Project00
    
"""

""" =======================  Import dependencies ========================== """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

""" =======================  Import DataSet ========================== """

plt.close('all') #close any open plots

def training(training_data):
    ground_truth = np.load('ground_truth.npy', 'r')
    data_train = training_data
    
    """ created a list where the RGB values in ground truth did not indicate the red color"""
    
    a = [3,13,23,26]
    
    """ Creating an array with 6250X6250X1 with zeros and storing the labels in third dimension
    if its red car - then it is labelled 1
    if its not a red car - then it is left as it is so it would be 0
    The X and Y coordinates of the location of the red car is copied into this new array"""
    
    label_data_train = np.zeros((6250,6250,1))
    for i in [x for x in range(0,28) if x not in a ]:
        first = ground_truth[i,1]
        second = ground_truth[i,0]
        label_data_train[first,second] = 1
         
    """ ============  Generate Training and validation Data =================== """
    #Creating lists for storing RGB values and Labels
    train_RGB = [] 
    Lab = [] 
    
    #Getting all the RGB values of the pixels in the train_RGB list and labels in the Lab list
    # Ideally the range of i and j should be (0,6250) and (0,6250) but to test a specific portion
    # of the image, certain pixels (850,970) and (4517,4985) are selected as i and j such that there are red cars in them
    for i in range(0,6250):
        for j in range(0,6250):
            train_RGB.append(data_train[i,j])
            
    for i in range(0,6250):
        for j in range(0,6250):
            Lab.append(label_data_train[i,j])
       
    # Here, Training data set X is the RGB values and y is the class labels
    """ ===================  Train and Cross Validation ======================= """
    X = train_RGB
    y1= np.array(Lab).reshape(-1,)
    y = y1
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, random_state=42)
    #KNN Classifier
    knn_ = KNeighborsClassifier(n_neighbors=1)
    knn_.fit(X_train, y_train)
    y_pred_class = knn_.predict(X_valid)
    print (metrics.accuracy_score(y_valid, y_pred_class))
    
    
    #KNN Classifier -- cross-validation to pick k
    Cv_knn = []
    Cv_kvalue = []
    Krange = np.arange(1,100,3)
    for k in Krange:   
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.4, random_state = 42)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions_KNN = knn.predict(X_valid)
        accuracy_KNN = metrics.accuracy_score(y_valid, predictions_KNN)
        Cv_kvalue.append(accuracy_KNN)
        Cv_knn.append(np.mean(Cv_kvalue))
    
    #Plot mean accuracy for different k
    plt.plot(Krange,Cv_knn)
    plt.ylabel('accuracy') 
    plt.xlabel('K')
    plt.title('Average accuracy for Different K')
    
    plt.show()
    
    return X, y, knn_;
    






