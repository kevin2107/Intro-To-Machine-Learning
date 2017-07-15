#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='rbf', C=10000)

features_train = features_train[:len(features_train)] 
labels_train = labels_train[:len(labels_train)] 

#Train
t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")
#predict
t0 = time()
pred = clf.predict(features_test)
print ("predicting time:", round(time()-t0, 3), "s")

accuracy = accuracy_score(pred, labels_test)
print ('accuracy = {0}'.format(accuracy))

x=0

for n in range(len(labels_train)):
	answer=pred[n]
	print (n, "Prediction =", answer)
	
	if answer == 1:
		x += 1
		print (x)
		
	
print ('total:',x)
#########################################################
