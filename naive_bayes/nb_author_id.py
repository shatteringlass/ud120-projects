#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
t0 = time.time()
cls = GaussianNB()

cls.fit(features_train, labels_test)
t1 = time.time()
time_to_fitness = t1-t0

accuracy = cls.accuracy_score(features_test, labels_test)
t2 = time.time()
print("Achieved accuracy {} in time {} ({} seconds to fit the classifier, {} seconds to test accuracy".format(accuracy, t2-t0, time_to_fitness, t2-t1))

return cls

#########################################################


