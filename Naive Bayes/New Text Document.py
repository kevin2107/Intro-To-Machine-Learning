from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
from ClassifyNB import classify

import numpy as np
import pylab as pl
import sys
from time import time
sys.path.append("../tools/")
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


##Linear Regression

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit ([[0, 2], [2, 2.3], [4, 2.7], [6, 2.9], [8, 3.4], [10, 3.6], [12,3.9]], [0, 1, 2, 3, 4, 5, 6])

reg.coef_

prettyPicture(reg, reg.fit, reg.fit)
output_image("test.png", "png", open("test.png", "rb").read())