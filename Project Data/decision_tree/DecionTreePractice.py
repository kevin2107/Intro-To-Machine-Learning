import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



########################## DECISION TREE #################################


### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively
x = features_train
y= labels_train

from sklearn import tree
from sklearn.metrics import accuracy_score
clf = tree.DecionTreeClassifier(min_samples_split=50)
clf = clf.fit(x, y)
pred = clf.fit(x, y)

acc = accuracy_score(pred, labels_test)

clf50 = tree.DecionTreeClassifier(min_samples_split=50)
clf50 = clf50.fit(x, y)
pred50 = clf50.fit(x, y)

acc = accuracy_score(pred, labels_test)
acc = accuracy_score(pred50, labels_test)

def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}