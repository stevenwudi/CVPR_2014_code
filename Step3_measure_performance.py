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
import sys, os, os.path,random,numpy,zipfile
from shutil import copyfile
import matplotlib.pyplot as plt

from ChalearnLAPEvaluation import evalGesture,exportGT_Gesture
from ChalearnLAPSample import GestureSample
from ChalearnLAPEvaluation import gesture_overlap_csv
from utils import Extract_feature

def main():

   prediction_dir =  r'I:\Kaggle_multimodal\StartingKit_track3\CoDaLab_Gesure_track3\matlab\prediction_650_conv'
   #prediction_dir =  r'I:\Kaggle_multimodal\StartingKit_track3\CoDaLab_Gesure_track3\matlab\prediction_650'
   #truth_dir = r'I:\Kaggle_multimodal\validation_labels'
   truth_dir = r'I:\Kaggle_multimodal\validation'
   final_score = evalGesture(prediction_dir,truth_dir)
   print "final_score "+str(final_score)

   # 3DCNN: final_score0.375025337775


if __name__ == '__main__':
    main()
