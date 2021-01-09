#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:48:08 2020

@author: tiantong
"""
#%% IMPORT PACKAGES
from scipy.io import loadmat
import numpy as np
from utils import *
#%% All the parameters

# These are folder path of Macbook
# BASE_FOLDER = '/Users/tiantong/Desktop/LMR/ICL_LMR/processedData/'
# SAVE_FOLDER = '/Users/tiantong/Desktop/LMR/ICL_LMR/featureAndLabel/'
# LW_FOLDER = 'LW/'
# RA_FOLDER = 'RA/'
# RD_FOLDER = 'RD/'

# These are the folder path of Win Laptop
BASE_FOLDER = 'E:/Work/LMR-CIL/processedData/'
SAVE_FOLDER = 'E:/Work/LMR-CIL/featureAndLabel/'
LW_FOLDER = 'LW/'
RA_FOLDER = 'RA/'
RD_FOLDER = 'RD/'

###############################################################################
# which degree you want to process
LW_DEGREE = [1] # 0 degree, level ground walking
RA_DEGREE = [i for i in range(3, 15)]
RD_DEGREE = [i for i in range(3, 15)]
# files per degree
nFilePerDegree = 10

# sliding window size and step size for feature extraction
WIN_SIZE = 50 # * 10 ms
STEP_SIZE = 1 # * 10 ms

# number of features
nFeature = 112
###############################################################################

#%% Load .mat data, feature extraction, and save features
label = 0
# level ground walking
foldPath = BASE_FOLDER + LW_FOLDER + '/'

for itemDegree in LW_DEGREE:
    totalFeatureMatrix = np.empty((0, nFeature))
    # -5 because I don't want too many samples, which results in LW has much more
    # samples than RA and RD.
    # total 5 files
    for j in range(nFilePerDegree-5): 
        jFile = j + 1 # index alignment
        # load into a dict
        dataDict = loadmat(foldPath+'Data_'+str(itemDegree)+'_'+str(jFile)+'.mat')
        
        # data that we are interested in
        imuShank = dataDict['IMU_Shank']
        imuThigh = dataDict['IMU_Thigh']
        heelStrike = dataDict['heel_strike']
        
        # concatenate two IMUs into a whole data matrix
        imuData = np.concatenate((imuShank, imuThigh), axis=1)
        
        # feature extraction for this file
        featureMatrix = featExtract(imuData, heelStrike, WIN_SIZE, STEP_SIZE, nFeature)
        totalFeatureMatrix = np.concatenate((totalFeatureMatrix, featureMatrix))
        
        message = 'LW: '+'degree_'+str(itemDegree)+' '+'file_'+str(jFile)
        print(message)
        
    # label this terrain
    labelVec = label * np.ones((totalFeatureMatrix.shape[0], ))
    print('label: '+str(label))
    
    
    # save this feature matrix and label
    np.save(SAVE_FOLDER+'class_'+str(label)+'_feature.npy', totalFeatureMatrix) 
    np.save(SAVE_FOLDER+'class_'+str(label)+'_label.npy', labelVec)
    
    # label increment
    label += 1

# ramp ascent
foldPath = BASE_FOLDER + RA_FOLDER + '/'

for itemDegree in RA_DEGREE:
    totalFeatureMatrix = np.empty((0, nFeature))
    for j in range(nFilePerDegree):
        jFile = j + 1 # index alignment
        # load into a dict
        dataDict = loadmat(foldPath+'Data_'+str(itemDegree)+'_'+str(jFile)+'.mat')
        
        # data that we are interested in
        imuShank = dataDict['IMU_Shank']
        imuThigh = dataDict['IMU_Thigh']
        heelStrike = dataDict['heel_strike']
        
        # concatenate two IMUs into a whole data matrix
        imuData = np.concatenate((imuShank, imuThigh), axis=1)
        
        # feature extraction for this file
        featureMatrix = featExtract(imuData, heelStrike, WIN_SIZE, STEP_SIZE, nFeature)
        totalFeatureMatrix = np.concatenate((totalFeatureMatrix, featureMatrix))
        
        message = 'RA: '+'degree_'+str(itemDegree)+' '+'file_'+str(jFile)
        print(message)
        
    # label this terrain
    labelVec = label * np.ones((totalFeatureMatrix.shape[0], ))
    print('label: '+str(label))
    
    
    # save this feature matrix and label
    np.save(SAVE_FOLDER+'class_'+str(label)+'_feature.npy', totalFeatureMatrix) 
    np.save(SAVE_FOLDER+'class_'+str(label)+'_label.npy', labelVec)
    
    # label increment
    label += 1

# ramp descent
foldPath = BASE_FOLDER + RD_FOLDER + '/'

for itemDegree in RA_DEGREE:
    totalFeatureMatrix = np.empty((0, nFeature))
    for j in range(nFilePerDegree):
        jFile = j + 1 # index alignment
        # load into a dict
        dataDict = loadmat(foldPath+'Data_'+str(itemDegree)+'_'+str(jFile)+'.mat')
        
        # data that we are interested in
        imuShank = dataDict['IMU_Shank']
        imuThigh = dataDict['IMU_Thigh']
        heelStrike = dataDict['heel_strike']
        
        # concatenate two IMUs into a whole data matrix
        imuData = np.concatenate((imuShank, imuThigh), axis=1)
        
        # feature extraction for this file
        featureMatrix = featExtract(imuData, heelStrike, WIN_SIZE, STEP_SIZE, nFeature)
        totalFeatureMatrix = np.concatenate((totalFeatureMatrix, featureMatrix))
        
        message = 'RD: '+'degree_'+str(itemDegree)+' '+'file_'+str(jFile)
        print(message)
        
    # label this terrain
    labelVec = label * np.ones((totalFeatureMatrix.shape[0], ))
    print('label: '+str(label))
    
    
    # save this feature matrix and label
    np.save(SAVE_FOLDER+'class_'+str(label)+'_feature.npy', totalFeatureMatrix) 
    np.save(SAVE_FOLDER+'class_'+str(label)+'_label.npy', labelVec)
    
    # label increment
    label += 1