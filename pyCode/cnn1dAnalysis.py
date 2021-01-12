#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 17:20:42 2021

@author: tiantong

Using 1D-CNN to extract high level features and then analize it.
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
import pickle
from utils import *

#%% All the parameters
nTimeStamp = 50
nChannel = 16

# load file path for MacOS
# LOAD_FOLDER = '/Users/tiantong/Desktop/LMR/ICL_LMR/tensorAndLabel/'

# load data path for windows
LOAD_FOLDER = 'E:/Work/LMR-CIL/tensorAndLabel/'
# save model path for windows
SAVE_FOLDER = 'E:/Work/LMR-CIL/trainedModel/'

# which classes you wanna use in training
# aka, base class
# CLASS_FOR_TRAINING = [1,5,10,13,17,22]
CLASS_FOR_TRAINING = [0,1,2,3,4,13,14,15,16]

# incremental class
# CLASS_FOR_INCREMENTAL = [0,3,8,12,15,20,24]
CLASS_FOR_INCREMENTAL = [7,9,12,19,21,24]
#%% Load the basic class data set
# All feature vecs in X participates in the training process
X = np.empty((0, nTimeStamp, nChannel))
y = np.empty((0, ))

for itemClass in CLASS_FOR_TRAINING:
    tempX = np.load(LOAD_FOLDER+'class_'+str(itemClass)+'_feature.npy')
    tempy = np.load(LOAD_FOLDER+'class_'+str(itemClass)+'_label.npy')
    X = np.concatenate((X, tempX), axis=0)
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
entireClass = CLASS_FOR_INCREMENTAL + CLASS_FOR_TRAINING

for itemClass in entireClass:
    tempX = np.load(LOAD_FOLDER+'class_'+str(itemClass)+'_feature.npy')
    tempy = np.load(LOAD_FOLDER+'class_'+str(itemClass)+'_label.npy')
    entireX = np.concatenate((entireX, tempX))
    entirey = np.concatenate((entirey, tempy))
#%% k folds cross validation

# number of folds
n_folds = 5

# cache
scores, histories, models, zscores= list(), list(), list(), list()


kfold = KFold(n_folds, shuffle=True, random_state=42)

for train_ix, test_ix in kfold.split(X):
	# define model
    model = defineCNNModel(nTimeStamp, nChannel, len(CLASS_FOR_TRAINING))
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
    print('Train on %d samples, test on %d samples.' %(trainX.shape[0], testX.shape[0]))
    history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1)
	# evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
	# stores logs
    scores.append(acc)
    histories.append(history)
    models.append(model)
    zscores.append(zscore)

# retrieve the model with the best val acc, and the corresponding scalar
bestModelIdx = scores.index(max(scores))
model = models[bestModelIdx]
zscore = zscores[bestModelIdx]
#%% Store the trained model

# store the trained model
model.save(SAVE_FOLDER + 'cnn1d.h5')
# store the standard scalar
pickle.dump(zscore, open(SAVE_FOLDER + 'cnn1d_scaler.pkl','wb'))
# zscore = pickle.load(open('scaler.pkl', 'rb'))
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

# # standardize entireX based on the training classes
# a, b, c = entireX.shape
# entireXFlatten = np.reshape(entireX, (a, b * c))
# entireXFlatten = zscore.fit_transform(entireXFlatten)
# entireX = np.reshape(entireXFlatten, (a, b, c))


# tsne = manifold.TSNE(n_components=2, init='pca', random_state=42)
# # peek at the last layer
# intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense1').output)
# X_peek = intermediate_layer_model.predict(entireX)
# # tsne transform
# X_tsne = tsne.fit_transform(X_peek)

# x_min, x_max = X_tsne.min(axis=0), X_tsne.max(axis=0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)  

# fig = plt.figure()
# ax = plt.gca()
# ax.scatter(X_tsne[:, 0], X_tsne[:, 1],s=20, marker='.', c=entirey)
# plt.show()