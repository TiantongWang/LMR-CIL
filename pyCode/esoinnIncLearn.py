# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:55:56 2021

@author: wang_

Incremental learning using ESOINN.
"""
#%% import packages
from utils import *
from esoinn import ESoinn
import pickle
import numpy as np
from keras.models import load_model, Model

#%% ALL THE PARAMETERS

nTimeStamp = 50
nChannel = 16

# load model path for windows
LOAD_MODEL_PATH = 'E:/Work/LMR-CIL/trainedModel/'

# load data path for windows
LOAD_DATA_PATH = 'E:/Work/LMR-CIL/tensorAndLabel/'

# incremental class
CLASS_FOR_INCREMENTAL = [8,12,20,24]

# hyper-parameters for esoinn
dim = nChannel
max_edge_age = 50
iteration_threshold = 200
c1 = 0.001
c2 = 1.0
#%% load the saved model and standardscalar
zscore = pickle.load(open(LOAD_MODEL_PATH + 'cnn1d_scaler.pkl', 'rb'))
model = load_model(LOAD_MODEL_PATH + 'cnn1d.h5')

#%% load the data for incremental learning

# load data set
X = np.empty((0, nTimeStamp, nChannel))
y = np.empty((0, )) 

# all the classes
entireClass = CLASS_FOR_INCREMENTAL
for itemClass in entireClass:
    tempX = np.load(LOAD_DATA_PATH+'class_'+str(itemClass)+'_feature.npy')
    tempy = np.load(LOAD_DATA_PATH+'class_'+str(itemClass)+'_label.npy')
    X = np.concatenate((X, tempX))
    y = np.concatenate((y, tempy))

#%% get the data after feature extractor

# standardize X based on the training classes
a, b, c = X.shape
XFlatten = np.reshape(X, (a, b * c))
XFlatten = zscore.fit_transform(XFlatten)
X = np.reshape(XFlatten, (a, b, c))

# data transformed by feature extractor
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense1').output)
X_peek = intermediate_layer_model.predict(X)

#%% incremental learning using esoinn
ESOINN = ESoinn(dim=dim, max_edge_age=max_edge_age, iteration_threshold=iteration_threshold,\
                c1=c1, c2=c2)
ESOINN.fit(X_peek)
