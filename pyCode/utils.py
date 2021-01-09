#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:50:03 2020

@author: tiantong
"""
#%% IMPORT PACKAGES
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import LSTM
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import SpatialDropout2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import TimeDistributed

#%% Functions
def computeWaveformLen(dataMatrix):
    '''
    This function computes the waveform length of each channel in dataMatrix.
    INPUTS:
        dataMatrix: a matrix of shape (m, n), m is number of samples, n is number of channels
    RETURNS：
        a vector that contains WL of each channel
    '''
    minuendMatrix = dataMatrix[1:, :]# 被减数
    subtrahendMatrix = dataMatrix[:-1, :] # 减数
    # difference
    diffMatrix = minuendMatrix - subtrahendMatrix
    # absolute value of difference
    absDiffMatrix = abs(diffMatrix)
    # waveform length
    waveformLenVec = np.sum(absDiffMatrix, axis=0)
    return waveformLenVec

def computeSkewness(dataMatrix):
    '''
    This function computes the skewness of each channel in dataMatrix.
    INPUTS:
        dataMatrix: a matrix of shape (m, n), m is number of samples, n is number of channels
    RETURNS：
        a vector that contains skewness of each channel
    '''
    # number of samples
    m = dataMatrix.shape[0]
    # mean of each channel
    meanVec = np.mean(dataMatrix, axis=0)
    # numerator (above)
    numerator = np.sum((dataMatrix - meanVec) ** 3, axis=0) / m
    # denominator (below)
    denominator = (np.sum((dataMatrix - meanVec) ** 2, axis=0) / m) ** 1.5
    
    skewness = numerator / denominator
    return skewness

def computeKurtosis(dataMatrix):
    '''
    This function computes the kurtosis of each channel in dataMatrix.
    INPUTS:
        dataMatrix: a matrix of shape (m, n), m is number of samples, n is number of channels
    RETURNS：
        a vector that contains kurtosis of each channel
    '''
    # number of samples
    m = dataMatrix.shape[0]
    # mean of each channel
    meanVec = np.mean(dataMatrix, axis=0)
    # numerator (above)
    numerator = np.sum((dataMatrix - meanVec) ** 4, axis=0) / m
    # denominator (below)
    denominator = (np.sum((dataMatrix - meanVec) ** 2, axis=0) / m) ** 2
    
    kurtosis = numerator / denominator
    return kurtosis

def winFeatExtract(data):
    '''
    This function extracts features within a windowed data.
    INPUTS:
        data: a matrix with the size of (m, n), m is the number of samples
        within a window, n is the number of features.
    RETURNS:
        a feature vector.
    '''
    # mean of each channel
    meanVec = np.mean(data, axis=0)
    # std of each channel
    stdVec = np.std(data, axis=0)
    # max of each channel
    maxVec = np.max(data, axis=0)
    # min of each channel
    minVec = np.min(data, axis=0)
    # waveform length of each channel
    waveFormLenVec = computeWaveformLen(data)
    # kurtosis of each channel
    kurtosis = computeKurtosis(data)
    # skewness of each channel
    skewness = computeSkewness(data)
    
    featureVec = np.concatenate((meanVec, stdVec, maxVec, minVec, \
                                 waveFormLenVec, kurtosis, skewness))
    return featureVec
    
def featExtract(data, heelStrike, WIN_SIZE, STEP_SIZE, nFeature):
    '''
    Given data, and heelStrike points, extract feature with window size of
    WIN_SIZE, and incremental of STEP_SIZE
    This function extracts features and slides window within each gaitcycle
    INPUTS:
        data: data matrix containing imuShank and imuThigh data, with the size
        of (m, n), m is the number of samples, n is the number of channels.
        heelStrike: an array of size of (nHeelStrike, 1), heel strike points.
        WIN_SIZE: sliding window size.
        STEP_SIZE: sliding window step.
        nFeature: total number of features that are extracted.
    RETURNS:
        a feature matrix with the size of (nWindow, nFeature)
    '''
    # number of gait cycles containing in data
    nGaitCycle = len(heelStrike) - 1
    featureMatrix = np.empty((0, nFeature))
    for iGait in range(nGaitCycle):
        # start point of this gait cycle
        gaitStart = heelStrike[iGait, 0]
        # end point of this gait cycle
        gaitEnd = heelStrike[iGait+1, 0] - 1
        # data chunk of this gait cycle
        gaitData = data[gaitStart: gaitEnd+1, :]
        # length of this gait cycle
        nGaitDataSample = gaitData.shape[0]
        for j in range(0, nGaitDataSample-WIN_SIZE, STEP_SIZE):
            gaitDataWindow = gaitData[j:j+WIN_SIZE, :]
            featureVec = winFeatExtract(gaitDataWindow)
            featureVec = featureVec.reshape(1, -1)
            
            featureMatrix = np.concatenate((featureMatrix, featureVec))
            
    return featureMatrix

def dataPartition(data, heelStrike, WIN_SIZE, STEP_SIZE):
    '''
    Given data, and heelStrike points,  patition the data with window size of
    WIN_SIZE, and incremental of STEP_SIZE
    This function patitions the data within each gaitcycle.
    INPUTS:
        data: data matrix containing imuShank and imuThigh data, with the size
        of (m, n), m is the number of samples, n is the number of channels.
        heelStrike: an array of size of (nHeelStrike, 1), heel strike points.
        WIN_SIZE: sliding window size.
        STEP_SIZE: sliding window step.
    RETURNS:
        a tensor with the size of (nSample, nTimeStamp, nChannel)
    '''
    # number of channels of data
    nChannel = data.shape[1]
    # number of gait cycles containing in data
    nGaitCycle = len(heelStrike) - 1
    tensor = np.empty((0, WIN_SIZE, nChannel))
    for iGait in range(nGaitCycle):
        # start point of this gait cycle
        gaitStart = heelStrike[iGait, 0]
        # end point of this gait cycle
        gaitEnd = heelStrike[iGait+1, 0] - 1
        # data chunk of this gait cycle
        gaitData = data[gaitStart: gaitEnd+1, :]
        # length of this gait cycle
        nGaitDataSample = gaitData.shape[0]
        for j in range(0, nGaitDataSample-WIN_SIZE, STEP_SIZE):
            gaitDataWindow = gaitData[j:j+WIN_SIZE, :]
            # reshape into a tensor, in order to concatenate
            gaitDataWindow = np.reshape(gaitDataWindow, (1, gaitDataWindow.shape[0], gaitDataWindow.shape[1]))
            tensor = np.concatenate((tensor, gaitDataWindow), axis=0)            
    return tensor

def defineBPModel(nFeature, nClass):
    '''
    This function defines a BP neural network architecture.
    INPUTS:
        nFeature: number of input features.
        nClass: the number of class you want to train.
    RETURNS:
        a compiled model.
    '''
    
    model = Sequential()
    
    # the first layer
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform', name='dense1', input_shape=(nFeature, )))
    # the second layer
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', name='dense2'))
    # the output layer
    model.add(Dense(nClass, activation='softmax'))
	# compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def defineCNNModel(nTimeStamp, nChannel, nClass):
    '''
    This function defines a CNN architecture.
    INPUTS:
        nTimeStamp: window size
        nChannel: number of channels
        nClass: the number of class you want to train.
    RETURNS:
        a compiled model.
    '''
    No 
    model = Sequential()
    # the first layer
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(nTimeStamp, nChannel)))
    # the second layer
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    # drop out
    model.add(Dropout(0.5))
    # max pool
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # the third layer
    model.add(Dense(50, activation='relu', name='dense1'))
    # output layer
    model.add(Dense(nClass, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def defineLSTMModel(nTimeStamp, nChannel, nClass):
    '''
    This function defines an LSTM architecture.
    INPUTS:
        nTimeStamp: window size, equals to time steps
        nChannel: number of channels, equals to number of features
        nClass: the number of class you want to train.
    RETURNS:
        a compiled model.
    '''
    model = Sequential()
    # first layer of LSTM
    model.add(LSTM(30, input_shape=(nTimeStamp, nChannel)))
    # dropout layer
    model.add(Dropout(0.5))
    # dense layer 1
    model.add(Dense(30, activation='relu', name='dense1'))
    # dense layer 2
    model.add(Dense(20, activation='relu', name='dense2'))
    # output layer
    model.add(Dense(nClass, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def defineCNNLSTMModel(nLength, nChannel, nClass):
    '''
    This function defines a CNN LSTM architecture.
    INPUTS:
        nLength: length of subsequences that are fed into time-distributed CNN
        nChannel: number of channels, equals to number of features
        nClass: the number of class you want to train.
    RETURNS:
        a compiled model.
    '''
    # define model
    model = Sequential()
    # time-distributed CNN layer 1
    model.add(TimeDistributed(Conv1D(filters=16, kernel_size=3, activation='relu'), input_shape=(None,nLength,nChannel)))
    # time-distributed CNN layer 2
    model.add(TimeDistributed(Conv1D(filters=16, kernel_size=3, activation='relu')))
    # time-distributed dropout
    model.add(TimeDistributed(Dropout(0.5)))
    # time-distributed pooling
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    # flatten
    model.add(TimeDistributed(Flatten()))
    # LSTM layer 1
    model.add(LSTM(20))
    # dropout
    model.add(Dropout(0.5))
    # fully connected layer 1
    model.add(Dense(20, activation='relu', name='dense1'))
    # output layer
    model.add(Dense(nClass, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def onehotToLabel(onehotMatrix):
    '''
    This function converts onehot-labeled matrix to an ordinary label vector.
    INPUTS:
        onehotMatrix: a matrix with the size of (nSample, nClass)
    RETURNS:
        an ordinary label vector with the size of (nSample, )       
    '''
    temp = np.nonzero(onehotMatrix)     
    labelVec = temp[1]
    return labelVec