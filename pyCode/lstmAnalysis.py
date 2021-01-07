#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:46:44 2021

@author: tiantong

Using LSTM to extract high level features and then analize it.
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
nTimeStamp = 50
nChannel = 16

LOAD_FOLDER = '/Users/tiantong/Desktop/LMR/ICL_LMR/tensorAndLabel/'

# which classes you wanna use in training
# aka, base class
#CLASS_FOR_TRAINING = [1,5,10,13,17,22]
CLASS_FOR_TRAINING = [0,1,2,3,4,13,14,15,16]

# incremental class
CLASS_FOR_INCREMENTAL = [5,6,7,8]

#%% Load the basic class data set
# All feature vecs in X participates in the training process
X = np.empty((0, nTimeStamp, nChannel))
y = np.empty((0, ))

for itemClass in CLASS_FOR_TRAINING:
    tempX = np.load(LOAD_FOLDER+'class_'+str(itemClass)+'_feature.npy')
    tempy = np.load(LOAD_FOLDER+'class_'+str(itemClass)+'_label.npy')
    X = np.concatenate((X, tempX), axis=0)
#    print(tempX.shape)
    y = np.concatenate((y, tempy))

# values in y are not continuous, however, when onehot encoding, the values in
# y have to be continuous, eg. [0, 1, 2, 3...] instead of [1, 4, 5, 7...]
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(y)
yOneHot = to_categorical(y, num_classes=len(CLASS_FOR_TRAINING))


# load the entire data set
entireX = np.empty((0, nTimeStamp, nChannel))
entirey = np.empty((0, )) 
# all the classes
##################### Modifiy It ! ############################################
entireClass = CLASS_FOR_INCREMENTAL
###############################################################################
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

# k-fold cross validation
kfold = KFold(n_folds, shuffle=True, random_state=42)

# split time steps for CNN-LSTM & CONV-LSTM


for train_ix, test_ix in kfold.split(X):
	# define model
    model = defineLSTMModel(nTimeStamp, nChannel, len(CLASS_FOR_TRAINING))
	# select rows for train and test
    trainX, trainY, testX, testY = X[train_ix], yOneHot[train_ix], X[test_ix], yOneHot[test_ix]
    # standardize trainX
    a, b, c = trainX.shape
    zscore = StandardScaler()
    trainXFlatten = np.reshape(trainX, (a, b * c))
    trainXFlatten = zscore.fit_transform(trainXFlatten)
    trainX = np.reshape(trainXFlatten, (a, b, c))
    # standardize testX
    a, b, c = testX.shape
    testXFlatten = np.reshape(testX, (a, b * c))
    testXFlatten = zscore.fit_transform(testXFlatten)
    testX = np.reshape(testXFlatten, (a, b, c))
	# fit model
    history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1)
	# evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
	# stores scores
    scores.append(acc)
    histories.append(history)
    models.append(model)

#%% last layer tsne visualization
##train on all classes, visualize on all classes
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=42)
## peek at the last layer
#intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense1').output)
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
##plt.show()

#%% train on base classes, visualize on all classes

# standardize entireX based on the training classes
a, b, c = entireX.shape
entireXFlatten = np.reshape(entireX, (a, b * c))
entireXFlatten = zscore.fit_transform(entireXFlatten)
entireX = np.reshape(entireXFlatten, (a, b, c))

# the layer name that you want to peek at.
layerName = 'dense2'
tsne = manifold.TSNE(n_components=2, init='pca', random_state=42)
# peek at the last layer
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layerName).output)
X_peek = intermediate_layer_model.predict(entireX)
# tsne transform
X_tsne = tsne.fit_transform(X_peek)

x_min, x_max = X_tsne.min(axis=0), X_tsne.max(axis=0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  

fig = plt.figure()
ax = plt.gca()
ax.scatter(X_tsne[:, 0], X_tsne[:, 1],s=0.5, marker='.', c=entirey)
plt.show()