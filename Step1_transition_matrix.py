#-------------------------------------------------------------------------------
# Name:        Starting Kit for ChaLearn LAP 2014 Track3
# Purpose:     Show basic functionality of provided code
#
# Author:      Xavier Baro
# Author:      Di Wu: stevenwudi@gmail.com
# Created:     24/03/2014
# Copyright:   (c) Chalearn LAP 2014
# Licence:     GPL3
#-------------------------------------------------------------------------------
import sys, os,random,numpy,zipfile
from shutil import copyfile
import matplotlib.pyplot as plt
import cv2
from ChalearnLAPEvaluation import evalGesture,exportGT_Gesture
from ChalearnLAPSample import GestureSample
from utils import IsLeftDominant
from utils import Extract_feature_normalized
from utils import Extract_feature
import time
import cPickle
""" Main script. Show how to perform all competition steps
    Access the sample information to learn a model. """
# Data folder (Training data)
print("Extracting the training files")
data=os.path.join("I:\Kaggle_multimodal\Training\\")  
# Get the list of training samples
samples=os.listdir(data)
used_joints = ['ElbowLeft', 'WristLeft', 'ElbowRight', 'WristRight']
njoints = len(used_joints)
STATE_NO = 10
batch_num = 13

# pre-allocating the memory
Prior = numpy.zeros(shape=(201))
Transition_matrix = numpy.zeros(shape=(201,201))

for file_count, file in enumerate(samples):
    #if not file.endswith(".zip"):
    #    continue;  
    time_tic = time.time()      
    if (file_count<651):
        print("\t Processing file " + file)
        # Create the object to access the sample
        smp=GestureSample(os.path.join(data,file))
        # ###############################################
        # USE Ground Truth information to learn the model
        # ###############################################
        # Get the list of actions for this frame
        gesturesList=smp.getGestures()


        for gesture in gesturesList:
            gestureID,startFrame,endFrame=gesture

            for frame in range(endFrame-startFrame+1-4):
                
                state_no_1 = numpy.floor(frame*(STATE_NO*1.0/(endFrame-startFrame+1-3)))
                state_no_1 = state_no_1+STATE_NO*(gestureID-1)
                state_no_2 = numpy.floor((frame+1)*(STATE_NO*1.0/(endFrame-startFrame+1-3)))
                state_no_2 = state_no_2+STATE_NO*(gestureID-1)
                ## we allow first two states add together:
                Prior [state_no_1] += 1
                Transition_matrix[state_no_1, state_no_2] += 1
                if frame<2:
                    Transition_matrix[-1, state_no_1] += 1
                    Prior[-1] += 1
                if frame> (endFrame-startFrame+1-4-2):
                    Transition_matrix[state_no_2, -1] += 1
                    Prior[-1] += 1
        del smp        

import scipy.io as sio
sio.savemat('Transition_matrix.mat', {'Transition_matrix':Transition_matrix})
sio.savemat('Prior.mat', {Prior:'Prior'})
sio.savemat('Prior_Transition_matrix.mat', {'Transition_matrix':Transition_matrix, 'Prior': Prior})

img = Transition_matrix*1.0*255/Transition_matrix.max()
fig, ax = plt.subplots()
cax = ax.imshow(temp2, interpolation='nearest', cmap=cm.coolwarm)
cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
cbar.ax.set_yticklabels(['< -1', '0', '> 1'])# vertically oriented colorbar

    