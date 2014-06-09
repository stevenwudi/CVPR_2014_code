#-------------------------------------------------------------------------------
# Name:        Chalearn LAP utils scripts
# Purpose:     Provide scripts to add labels to Chalearn LAP challenge tracks samples
#
# Author:      Xavier Baro
#              Di Wu: stevenwudi@gmail.com
# Created:     25/04/2014
# Copyright:   (c) Chalearn LAP 2014
# Licence:     GPL
#-------------------------------------------------------------------------------
import os
import zipfile
import shutil
import glob

def main():
    """ Main script. Created a labeled copy of validation samples """
    # Data folder (Unlabeled data samples)
    dataPath=r'I:\Kaggle_multimodal\Validation'
    # Labels file (Unziped validation.zip)
    labelsPath=r'I:\Kaggle_multimodal\validation_labels'    
    # Use the method for desired track
    print('Uncoment the line for your track')
    addLabels_Track3(dataPath, labelsPath)

    
def addLabels_Track3(dataPath, labelsPath):
    """ Add labels to the samples"""   
    # Check the given data path
    if not os.path.exists(dataPath) or not os.path.isdir(dataPath):
        raise Exception("Data path does not exist: " + dataPath)    
    # Check the given labels path
    if not os.path.exists(labelsPath) or not os.path.isdir(labelsPath):
        raise Exception("Labels path does not exist: " + labelsPath)    

    # Get the list of samples
    samplesList = os.listdir(dataPath)
    # For each sample on the GT, search the given prediction
    for sample in samplesList:
        print "writing file" + sample
        # Build paths for sample
        sampleFile = os.path.join(dataPath, sample)
        # Prepare sample information
        file = os.path.split(sampleFile)[1]
        sampleID = os.path.splitext(file)[0]
        samplePath = dataPath + os.path.sep + sampleID

            # Add the labels
        srtFileName=sampleID + '_labels.csv'
        srcSampleDataPath = os.path.join(labelsPath, srtFileName)
        dstSampleDataPath = os.path.join(sampleFile, srtFileName)
        shutil.copyfile(srcSampleDataPath, dstSampleDataPath)


if __name__ == '__main__':
    main()
