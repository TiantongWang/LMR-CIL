#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 13:54:49 2020

@author: tiantong

Using BPNN to extract high level features and then analize it.
"""

#%% IMPORT PACKAGES
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn import manifold
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from mpl_toolkits.mplot3d import Axes3D
from utils import *

#%% All the parameters
# number of features
nFeature = 112

# LOAD_FOLDER = '/Users/tiantong/Desktop/LMR/ICL_LMR/featureAndLabel/'
LOAD_FOLDER = 'E:\\Work\\LMR-CIL\\featureAndLabel\\'

# which classes you wanna use in training
# aka, base class
#CLASS_FOR_TRAINING = [1,5,10,13,17,22]
CLASS_FOR_TRAINING = [0,1,2,3,4,13,14,15,16]

# incremental class
#CLASS_FOR_INCREMENTAL = [0,3,8,12,15,20,24]
CLASS_FOR_INCREMENTAL = [7,11,14]

#%% Load the basic class data set
# All feature vecs in X participates in the training process
X = np.empty((0, nFeature))
y = np.empty((0, ))

for itemClass in CLASS_FOR_TRAINING:
    tempX = np.load(LOAD_FOLDER+'class_'+str(itemClass)+'_feature.npy')
    tempy = np.load(LOAD_FOLDER+'class_'+str(itemClass)+'_label.npy')
    X = np.concatenate((X, tempX))
    y = np.concatenate((y, tempy))

# values in y are not continuous, however, when onehot encoding, the values in
# y have to be continuous.
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(y)
yOneHot = to_categorical(y, num_classes=len(CLASS_FOR_TRAINING))


# load the entire data set
entireX = np.empty((0, nFeature))
entirey = np.empty((0, )) 
# all the classes
entireClass = CLASS_FOR_INCREMENTAL

for itemClass in entireClass:
    tempX = np.load(LOAD_FOLDER+'class_'+str(itemClass)+'_feature.npy')
    tempy = np.load(LOAD_FOLDER+'class_'+str(itemClass)+'_label.npy')
    entireX = np.concatenate((entireX, tempX))
    entirey = np.concatenate((entirey, tempy))
#%% k folds cross validation

# number of folds
n_folds = 5

# cache
scores, histories, models= list(), list(), list()


kfold = KFold(n_folds, shuffle=True, random_state=42)

for train_ix, test_ix in kfold.split(X):
	# define model
    model = defineBPModel(nFeature, len(CLASS_FOR_TRAINING))
	# select rows for train and test
    trainX, trainY, testX, testY = X[train_ix], yOneHot[train_ix], X[test_ix], yOneHot[test_ix]
    # standardize trainX
    zscore = StandardScaler()
    trainX = zscore.fit_transform(trainX)
    # standardize testX
    testX = zscore.transform(testX)
	# fit model
    print('Train on %d samples, test on %d samples.' %(trainX.shape[0], testX.shape[0]))
    history = model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(testX, testY), verbose=1)
	# evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
	# stores scores
    scores.append(acc)
    histories.append(history)
    models.append(model)

#%% last layer tsne visualization
# train on all classes, visualize on all classes

#tsne = manifold.TSNE(n_components=2, init='pca', random_state=42)
## peek at the last layer
#intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense2').output)
#X_peek = intermediate_layer_model.predict(testX)
## tsne transform
#X_tsne = tsne.fit_transform(X_peek)
#
#x_min, x_max = X_tsne.min(axis=0), X_tsne.max(axis=0)
#X_norm = (X_tsne - x_min) / (x_max - x_min)  
#
#label = onehotToLabel(testY)
#ax = plt.gca()
#ax.scatter(X_tsne[:, 0], X_tsne[:, 1],marker='.', c=label)
#plt.show()

#%% train on base classes, visualize on all classes

# standardize entireX based on the training classes
entireX = zscore.transform(entireX)
tsne = manifold.TSNE(n_components=2, init='pca', random_state=42)
# peek at the last layer
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense2').output)
X_peek = intermediate_layer_model.predict(entireX)
# tsne transform
X_tsne = tsne.fit_transform(X_peek)

x_min, x_max = X_tsne.min(axis=0), X_tsne.max(axis=0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  

fig = plt.figure()
ax = plt.gca()
ax.scatter(X_tsne[:, 0], X_tsne[:, 1],s=5, marker='.', c=entirey)
plt.show()

#fig = plt.figure()
#ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
#ax.scatter(X_norm[:, 0], X_norm[:, 1], X_norm[:, 2],s=1, marker='.',c=entirey)