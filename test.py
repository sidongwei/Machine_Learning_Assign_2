import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
X = np.array(pd.read_csv("MNIST_Xtrain.csv",header=None))   #using array for easier reshape
y = np.array(pd.read_csv("MNIST_ytrain.csv",header=None))
X_test = np.array(pd.read_csv("MNIST_Xtestp.csv",header=None))

#The following code of deskew refers to https://fsix.github.io/mnist/Deskewing.html
#Give credit to Dibya Ghosh and Alvin Wan

from scipy.ndimage import interpolation
def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix
def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)
def deskewAll(X):
    currents = []
    for i in range(len(X)):
        currents.append(deskew(X[i].reshape(28,28).T).flatten())
    return np.array(currents)

X_deskewed = deskewAll(X)
X_test_deskewed = deskewAll(X_test)

'''
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
'''

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3,n_jobs=-1)

model.fit(X_deskewed,y.reshape(60000))

y_pred = model.predict(X_test_deskewed)

n = len(y_pred)
ID = (np.array(range(n))+1)

#data = np.concatenate((ID,y_pred.astype(int)),axis=1)
result = pd.DataFrame({"ImageID":ID,"Digit":y_pred.astype(int).reshape(n)})
result.to_csv("20xxxx06.csv",index=False)
