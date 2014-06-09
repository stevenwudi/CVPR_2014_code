#-------------------------------------------------------------------------------
# Name:        Chalearn LAP sample
# Purpose:     Provide easy access to Chalearn LAP challenge data samples
#
# Author:      Xavier Baro
#
# Created:     21/01/2014
# Copyright:   (c) Xavier Baro 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os
import zipfile
import shutil
import cv2
import numpy
import csv
from PIL import Image, ImageDraw
from scipy.misc import imresize


class Skeleton(object):
    """ Class that represents the skeleton information """
    #define a class to encode skeleton data
    def __init__(self,data):
        """ Constructor. Reads skeleton information from given raw data """
        # Create an object from raw data
        self.joins=dict();
        pos=0
        self.joins['HipCenter']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['Spine']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ShoulderCenter']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['Head']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ShoulderLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ElbowLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['WristLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HandLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ShoulderRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ElbowRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['WristRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HandRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HipLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['KneeLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['AnkleLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['FootLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HipRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['KneeRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['AnkleRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['FootRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
    def getAllData(self):
        """ Return a dictionary with all the information for each skeleton node """
        return self.joins
    def getWorldCoordinates(self):
        """ Get World coordinates for each skeleton node """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][0]
        return skel
    def getJoinOrientations(self):
        """ Get orientations of all skeleton nodes """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][1]
        return skel
    def getPixelCoordinates(self):
        """ Get Pixel coordinates for each skeleton node """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][2]
        return skel
    def toImage(self,width,height,bgColor):
        """ Create an image for the skeleton information """
        SkeletonConnectionMap = (['HipCenter','Spine'],['Spine','ShoulderCenter'],['ShoulderCenter','Head'],['ShoulderCenter','ShoulderLeft'], \
                                 ['ShoulderLeft','ElbowLeft'],['ElbowLeft','WristLeft'],['WristLeft','HandLeft'],['ShoulderCenter','ShoulderRight'], \
                                 ['ShoulderRight','ElbowRight'],['ElbowRight','WristRight'],['WristRight','HandRight'],['HipCenter','HipRight'], \
                                 ['HipRight','KneeRight'],['KneeRight','AnkleRight'],['AnkleRight','FootRight'],['HipCenter','HipLeft'], \
                                 ['HipLeft','KneeLeft'],['KneeLeft','AnkleLeft'],['AnkleLeft','FootLeft'])
        im = Image.new('RGB', (width, height), bgColor)
        draw = ImageDraw.Draw(im)
        for link in SkeletonConnectionMap:
            p=self.getPixelCoordinates()[link[1]]
            p.extend(self.getPixelCoordinates()[link[0]])
            draw.line(p, fill=(255,0,0), width=5)
        for node in self.getPixelCoordinates().keys():
            p=self.getPixelCoordinates()[node]
            r=5
            draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=(0,0,255))
        del draw
        image = numpy.array(im)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image


class GestureSample(object):
    """ Class that allows to access all the information for a certain gesture database sample """
    #define class to access gesture data samples
    def __init__ (self,fileName):
        """ Constructor. Read the sample file and unzip it if it is necessary. All the data is loaded.

            sample=GestureSample('Sample0001.zip')

        """
        # Check the given file
        if not os.path.exists(fileName): #or not os.path.isfile(fileName):
            raise Exception("Sample path does not exist: " + fileName)

        # Prepare sample information
        self.fullFile = fileName
        self.dataPath = os.path.split(fileName)[0]
        self.file=os.path.split(fileName)[1]
        self.seqID=os.path.splitext(self.file)[0]
        self.samplePath=self.dataPath + os.path.sep + self.seqID;

        # Unzip sample if it is necessary
        if os.path.isdir(self.samplePath) :
            self.unzip = False
        else:
            self.unzip = True
            zipFile=zipfile.ZipFile(self.fullFile,"r")
            zipFile.extractall(self.samplePath)

        # Open video access for RGB information
        rgbVideoPath=self.samplePath + os.path.sep + self.seqID + '_color.mp4'
        if not os.path.exists(rgbVideoPath):
            raise Exception("Invalid sample file. RGB data is not available")
        self.rgb = cv2.VideoCapture(rgbVideoPath)
        while not self.rgb.isOpened():
            self.rgb = cv2.VideoCapture(rgbVideoPath)
            cv2.waitKey(500)
            # Open video access for Depth information
        depthVideoPath=self.samplePath + os.path.sep + self.seqID + '_depth.mp4'
        if not os.path.exists(depthVideoPath):
            raise Exception("Invalid sample file. Depth data is not available")
        self.depth = cv2.VideoCapture(depthVideoPath)
        while not self.depth.isOpened():
            self.depth = cv2.VideoCapture(depthVideoPath)
            cv2.waitKey(500)
            # Open video access for User segmentation information
        userVideoPath=self.samplePath + os.path.sep + self.seqID + '_user.mp4'
        if not os.path.exists(userVideoPath):
            raise Exception("Invalid sample file. User segmentation data is not available")
        self.user = cv2.VideoCapture(userVideoPath)
        while not self.user.isOpened():
            self.user = cv2.VideoCapture(userVideoPath)
            cv2.waitKey(500)
            # Read skeleton data
        skeletonPath=self.samplePath + os.path.sep + self.seqID + '_skeleton.csv'
        if not os.path.exists(skeletonPath):
            raise Exception("Invalid sample file. Skeleton data is not available")
        self.skeletons=[]
        with open(skeletonPath, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                self.skeletons.append(Skeleton(row))
            del filereader
            # Read sample data
        sampleDataPath=self.samplePath + os.path.sep + self.seqID + '_data.csv'
        if not os.path.exists(sampleDataPath):
            raise Exception("Invalid sample file. Sample data is not available")
        self.data=dict()
        with open(sampleDataPath, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                self.data['numFrames']=int(row[0])
                self.data['fps']=int(row[1])
                self.data['maxDepth']=int(row[2])
            del filereader
            # Read labels data
        labelsPath=self.samplePath + os.path.sep + self.seqID + '_labels.csv'
        if not os.path.exists(labelsPath):
            #warnings.warn("Labels are not available", Warning)
            self.labels=[]           
        else:
            self.labels=[]
            with open(labelsPath, 'rb') as csvfile:
                filereader = csv.reader(csvfile, delimiter=',')
                for row in filereader:
                    self.labels.append(map(int,row))
                del filereader
    def __del__(self):
        """ Destructor. If the object unziped the sample, it remove the temporal data """
        if self.unzip:
            self.clean()
    def clean(self):
        """ Clean temporal unziped data """
        del self.rgb;
        del self.depth;
        del self.user;
        shutil.rmtree(self.samplePath)
    def getFrame(self,video, frameNum):
        """ Get a single frame from given video object """
        # Check frame number
        # Get total number of frames
        numFrames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        # Check the given file
        if frameNum<1 or frameNum>numFrames:
            raise Exception("Invalid frame number <" + str(frameNum) + ">. Valid frames are values between 1 and " + str(int(numFrames)))
            # Set the frame index
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frameNum-1)
        ret,frame=video.read()
        if ret==False:
            raise Exception("Cannot read the frame")
        return frame
    def getRGB(self, frameNum):
        """ Get the RGB color image for the given frame """
        #get RGB frame
        return self.getFrame(self.rgb,frameNum)
    def getDepth(self, frameNum):
        """ Get the depth image for the given frame """
        #get Depth frame
        depthData=self.getFrame(self.depth,frameNum)
        # Convert to grayscale
        depthGray=cv2.cvtColor(depthData,cv2.cv.CV_RGB2GRAY)
        # Convert to float point
        depth=depthGray.astype(numpy.float32)
        # Convert to depth values
        depth=depth/255.0*float(self.data['maxDepth'])
        depth=depth.round()
        depth=depth.astype(numpy.uint16)
        return depth
    def getUser(self, frameNum):
        """ Get user segmentation image for the given frame """
        #get user segmentation frame
        return self.getFrame(self.user,frameNum)
    def getSkeleton(self, frameNum):
        """ Get the skeleton information for a given frame. It returns a Skeleton object """
        #get user skeleton for a given frame
        # Check frame number
        # Get total number of frames
        numFrames = len(self.skeletons)
        # Check the given file
        if frameNum<1 or frameNum>numFrames:
            raise Exception("Invalid frame number <" + str(frameNum) + ">. Valid frames are values between 1 and " + str(int(numFrames)))
        return self.skeletons[frameNum-1]
    def getSkeletonImage(self, frameNum):
        """ Create an image with the skeleton image for a given frame """
        return self.getSkeleton(frameNum).toImage(640,480,(255,255,255))

    def getNumFrames(self):
        """ Get the number of frames for this sample """
        return self.data['numFrames']

    def getComposedFrame(self, frameNum):
        """ Get a composition of all the modalities for a given frame """
        # get sample modalities
        rgb=self.getRGB(frameNum)
        depthValues=self.getDepth(frameNum)
        user=self.getUser(frameNum)
        skel=self.getSkeletonImage(frameNum)

        # Build depth image
        depth = depthValues.astype(numpy.float32)
        depth = depth*255.0/float(self.data['maxDepth'])
        depth = depth.round()
        depth = depth.astype(numpy.uint8)
        depth = cv2.applyColorMap(depth,cv2.COLORMAP_JET)

        # Build final image
        compSize1=(max(rgb.shape[0],depth.shape[0]),rgb.shape[1]+depth.shape[1])
        compSize2=(max(user.shape[0],skel.shape[0]),user.shape[1]+skel.shape[1])
        comp = numpy.zeros((compSize1[0]+ compSize2[0],max(compSize1[1],compSize2[1]),3), numpy.uint8)

        # Create composition
        comp[:rgb.shape[0],:rgb.shape[1],:]=rgb
        comp[:depth.shape[0],rgb.shape[1]:rgb.shape[1]+depth.shape[1],:]=depth
        comp[compSize1[0]:compSize1[0]+user.shape[0],:user.shape[1],:]=user
        comp[compSize1[0]:compSize1[0]+skel.shape[0],user.shape[1]:user.shape[1]+skel.shape[1],:]=skel

        return comp

    def getComposedFrameOverlapUser(self, frameNum):
        """ Get a composition of all the modalities for a given frame """
        # get sample modalities
        rgb=self.getRGB(frameNum)
        depthValues=self.getDepth(frameNum)
        user=self.getUser(frameNum)
        mask = numpy.mean(user, axis=2) > 150
        mask = numpy.tile(mask, (3,1,1))
        mask = mask.transpose((1,2,0))

        # Build depth image
        depth = depthValues.astype(numpy.float32)
        depth = depth*255.0/float(self.data['maxDepth'])
        depth = depth.round()
        depth = depth.astype(numpy.uint8)
        depth = cv2.applyColorMap(depth,cv2.COLORMAP_JET)

        # Build final image
        compSize=(max(rgb.shape[0],depth.shape[0]),rgb.shape[1]+depth.shape[1])
        comp = numpy.zeros((compSize[0]+ compSize[0],max(compSize[1],compSize[1]),3), numpy.uint8)

        # Create composition
        comp[:rgb.shape[0],:rgb.shape[1],:]=rgb
        comp[:depth.shape[0],rgb.shape[1]:rgb.shape[1]+depth.shape[1],:]= depth
        comp[compSize[0]:compSize[0]+user.shape[0],:user.shape[1],:]= mask * rgb
        comp[compSize[0]:compSize[0]+user.shape[0],user.shape[1]:user.shape[1]+user.shape[1],:]= mask * depth

        return comp

    def getComposedFrame_480(self, frameNum, ratio=0.5, topCut=60, botCut=140):
        """ Get a composition of all the modalities for a given frame """
        # get sample modalities
       
        rgb=self.getRGB(frameNum)
        rgb = rgb[topCut:-topCut,botCut:-botCut,:]
        
        rgb = imresize(rgb, ratio, interp='bilinear')
        depthValues=self.getDepth(frameNum)
        user=self.getUser(frameNum)
        user = user[topCut:-topCut,botCut:-botCut,:]
        user = imresize(user, ratio, interp='bilinear')
        mask = numpy.mean(user, axis=2) > 150
        mask = numpy.tile(mask, (3,1,1))
        mask = mask.transpose((1,2,0))

        # Build depth image
        depth = depthValues.astype(numpy.float32)
        depth = depth*255.0/float(self.data['maxDepth'])
        depth = depth.round()
        depth = depth[topCut:-topCut,botCut:-botCut]
        depth = imresize(depth, ratio, interp='bilinear')
        depth = depth.astype(numpy.uint8)
        depth = cv2.applyColorMap(depth,cv2.COLORMAP_JET)

        # Build final image
        compSize=(max(rgb.shape[0],depth.shape[0]),rgb.shape[1]+depth.shape[1])
        comp = numpy.zeros((compSize[0]+ compSize[0],max(compSize[1],compSize[1]),3), numpy.uint8)

        # Create composition
        comp[:rgb.shape[0],:rgb.shape[1],:]=rgb
        comp[:depth.shape[0],rgb.shape[1]:rgb.shape[1]+depth.shape[1],:]= depth
        comp[compSize[0]:compSize[0]+user.shape[0],:user.shape[1],:]= mask * rgb
        comp[compSize[0]:compSize[0]+user.shape[0],user.shape[1]:user.shape[1]+user.shape[1],:]= mask * depth

        return comp

    def getDepth3DCNN(self, frameNum, ratio=0.5, topCut=60, botCut=140):
        """ Get a composition of all the modalities for a given frame """
        # get sample modalities
       
        depthValues=self.getDepth(frameNum)
        user=self.getUser(frameNum)
        user = user[topCut:-topCut,botCut:-botCut,:]
        user = imresize(user, ratio, interp='bilinear')
        mask = numpy.mean(user, axis=2) > 150


        # Build depth image
        depth = depthValues.astype(numpy.float32)
        depth = depth*255.0/float(self.data['maxDepth'])
        depth = depth.round()
        depth = depth[topCut:-topCut,botCut:-botCut]
        depth = imresize(depth, ratio, interp='bilinear')
        depth = depth.astype(numpy.uint8)

        return  mask * depth

    def getDepthOverlapUser(self, frameNum, x_centre, y_centre, pixel_value, extractedFrameSize=224, upshift = 0):
        """ Get a composition of all the modalities for a given frame """
        halfFrameSize = extractedFrameSize/2
        user=self.getUser(frameNum)
        mask = numpy.mean(user, axis=2) > 150

        ratio = pixel_value/ 3000

        # Build depth image
        # get sample modalities
        depthValues=self.getDepth(frameNum)
        depth = depthValues.astype(numpy.float32)        
        depth = depth*255.0/float(self.data['maxDepth'])

        mask = imresize(mask, ratio, interp='nearest')
        depth = imresize(depth, ratio, interp='bilinear')

        depth_temp = depth * mask
        depth_extracted = depth_temp[x_centre-halfFrameSize-upshift:x_centre+halfFrameSize-upshift, y_centre-halfFrameSize: y_centre+halfFrameSize]

        depth = depth.round()
        depth = depth.astype(numpy.uint8)
        depth = cv2.applyColorMap(depth,cv2.COLORMAP_JET)

        depth_extracted = depth_extracted.round()
        depth_extracted = depth_extracted.astype(numpy.uint8)
        depth_extracted = cv2.applyColorMap(depth_extracted,cv2.COLORMAP_JET)
        

        # Build final image
        compSize=(depth.shape[0],depth.shape[1])
        comp = numpy.zeros((compSize[0] + extractedFrameSize,compSize[1]+compSize[1],3), numpy.uint8)

        # Create composition
        comp[:depth.shape[0],:depth.shape[1],:]=depth
        mask_new = numpy.tile(mask, (3,1,1))
        mask_new = mask_new.transpose((1,2,0))
        comp[:depth.shape[0],depth.shape[1]:depth.shape[1]+depth.shape[1],:]= mask_new * depth
        comp[compSize[0]:,:extractedFrameSize,:]= depth_extracted

        return comp

    def getDepthCentroid(self, startFrame, endFrame):
        """ Get a composition of all the modalities for a given frame """
        x_centre = []
        y_centre = []
        pixel_value = []

        for frameNum in range(startFrame, endFrame):
            user=self.getUser(frameNum)
            depthValues=self.getDepth(frameNum)
            depth = depthValues.astype(numpy.float32)
            #depth = depth*255.0/float(self.data['maxDepth'])
            mask = numpy.mean(user, axis=2) > 150
            width, height = mask.shape
            XX, YY, count, pixel_sum = 0, 0, 0, 0
            for x in range(width):
                for y in range(height):
                    if mask[x, y]:
                        XX += x
                        YY += y
                        count += 1
                        pixel_sum += depth[x, y]
            if count>0:
                x_centre.append(XX/count)
                y_centre.append(YY/count)
                pixel_value.append(pixel_sum/count)

        return [numpy.mean(x_centre), numpy.mean(y_centre), numpy.mean(pixel_value)]

    def getGestures(self):
        """ Get the list of gesture for this sample. Each row is a gesture, with the format (gestureID,startFrame,endFrame) """
        return self.labels
    def getGestureName(self,gestureID):
        """ Get the gesture label from a given gesture ID """
        names=('vattene','vieniqui','perfetto','furbo','cheduepalle','chevuoi','daccordo','seipazzo', \
               'combinato','freganiente','ok','cosatifarei','basta','prendere','noncenepiu','fame','tantotempo', \
               'buonissimo','messidaccordo','sonostufo')
        # Check the given file
        if gestureID<1 or gestureID>20:
            raise Exception("Invalid gesture ID <" + str(gestureID) + ">. Valid IDs are values between 1 and 20")
        return names[gestureID-1]

    def exportPredictions(self, prediction,predPath):
        """ Export the given prediction to the correct file in the given predictions path """
        if not os.path.exists(predPath):
            os.makedirs(predPath)
        output_filename = os.path.join(predPath,  self.seqID + '_prediction.csv')
        output_file = open(output_filename, 'wb')
        for row in prediction:
            output_file.write(repr(int(row[0])) + "," + repr(int(row[1])) + "," + repr(int(row[2])) + "\n")
        output_file.close()

    def play_video(self):
        """ 
        play the video, Wudi adds this
        """
        # Open video access for RGB information
        rgbVideoPath=self.samplePath + os.path.sep + self.seqID + '_color.mp4'
        if not os.path.exists(rgbVideoPath):
            raise Exception("Invalid sample file. RGB data is not available")

        self.rgb = cv2.VideoCapture(rgbVideoPath)
        while (self.rgb.isOpened()):
            ret, frame = self.rgb.read()
            cv2.imshow('frame',frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        self.rgb.release()
        cv2.destroyAllWindows()



    def evaluate(self,csvpathpred):
        """ Evaluate this sample agains the ground truth file """
        maxGestures=11
        seqLength=self.getNumFrames()

        # Get the list of gestures from the ground truth and frame activation
        predGestures = []
        binvec_pred = numpy.zeros((maxGestures, seqLength))
        gtGestures = []
        binvec_gt = numpy.zeros((maxGestures, seqLength))
        with open(csvpathpred, 'rb') as csvfilegt:
            csvgt = csv.reader(csvfilegt)
            for row in csvgt:
                binvec_pred[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
                predGestures.append(int(row[0]))

        # Get the list of gestures from prediction and frame activation
        for row in self.getActions():
                binvec_gt[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
                gtGestures.append(int(row[0]))

        # Get the list of gestures without repetitions for ground truth and predicton
        gtGestures = numpy.unique(gtGestures)
        predGestures = numpy.unique(predGestures)

        # Find false positives
        falsePos=numpy.setdiff1d(gtGestures, numpy.union1d(gtGestures,predGestures))

        # Get overlaps for each gesture
        overlaps = []
        for idx in gtGestures:
            intersec = sum(binvec_gt[idx-1] * binvec_pred[idx-1])
            aux = binvec_gt[idx-1] + binvec_pred[idx-1]
            union = sum(aux > 0)
            overlaps.append(intersec/union)

        # Use real gestures and false positive gestures to calculate the final score
        return sum(overlaps)/(len(overlaps)+len(falsePos))

    def get_shift_scale(self, template, ref_depth, start_frame=10, end_frame=20, debug_show=False):
        """
        Wudi add this method for extracting normalizing depth wrt Sample0003
        """
        from skimage.feature import match_template
        Feature_all =  numpy.zeros(shape=(480, 640,  end_frame-start_frame), dtype=numpy.uint16 )
        count = 0
        for frame_num in range(start_frame,end_frame):
                depth_original = self.getDepth(frame_num)
                mask = numpy.mean(self.getUser(frame_num), axis=2) > 150
                Feature_all[:, :, count] = depth_original * mask
                count += 1

        depth_image = Feature_all.mean(axis = 2)
        depth_image_normalized = depth_image * 1.0 / float(self.data['maxDepth'])
        depth_image_normalized /= depth_image_normalized.max()
        result = match_template(depth_image_normalized, template, pad_input=True)
        #############plot
        x, y = numpy.unravel_index(numpy.argmax(result), result.shape)
        shift = [depth_image.shape[0]/2-x, depth_image.shape[1]/2-y]

        subsize = 25 # we use 25 by 25 region as a measurement for median of distance
        minX = max(x - subsize,0)
        minY = max(y - subsize,0)
        maxX = min(x + subsize,depth_image.shape[0])
        maxY = min(y + subsize,depth_image.shape[1])
        subregion = depth_image[minX:maxX, minY:maxY]
        distance = numpy.median(subregion[subregion>0])
        scaling = distance*1.0 / ref_depth

        from matplotlib import pyplot as plt
        print "[x, y, shift, distance, scaling]"
        print str([x, y, shift, distance, scaling])

        if debug_show:           
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(8, 4))
            ax1.imshow(template)
            ax1.set_axis_off()
            ax1.set_title('template')

            ax2.imshow(depth_image_normalized)
            ax2.set_axis_off()
            ax2.set_title('image')
            # highlight matched region
            hcoin, wcoin = template.shape
            rect = plt.Rectangle((y-hcoin/2, x-wcoin/2), wcoin, hcoin, edgecolor='r', facecolor='none')
            ax2.add_patch(rect)


            import cv2
            from scipy.misc import imresize
            rows,cols = depth_image_normalized.shape
            M = numpy.float32([[1,0, shift[1]],[0,1, shift[0]]])
            affine_image = cv2.warpAffine(depth_image_normalized, M, (cols, rows))
            resize_image = imresize(affine_image, scaling)
            resize_image_median = cv2.medianBlur(resize_image,5)

            ax3.imshow(resize_image_median)
            ax3.set_axis_off()
            ax3.set_title('image_transformed')
            # highlight matched region
            hcoin, wcoin = resize_image_median.shape
            rect = plt.Rectangle((wcoin/2-160, hcoin/2-160), 320, 320, edgecolor='r', facecolor='none')
            ax3.add_patch(rect)

            ax4.imshow(result)
            ax4.set_axis_off()
            ax4.set_title('`match_template`\nresult')
            # highlight matched region
            ax4.autoscale(False)
            ax4.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

            plt.show()

        return [shift, scaling]


    def get_shift_scale_depth(self, shift, scale, framenumber, IM_SZ, show_flag=False):
        """
        Wudi added this method to extract segmented depth frame,
        by a shift and scale

        """
        depth_original = self.getDepth(framenumber)
        mask = numpy.mean(self.getUser(framenumber), axis=2) > 150
        resize_final_out = numpy.zeros((IM_SZ,IM_SZ))
        if mask.sum() < 1000: # Kinect detect nothing
            print "skip "+ str(framenumber)
            flag = False
        else:
            flag = True
            depth_user = depth_original * mask

            depth_user_normalized = depth_user * 1.0 / float(self.data['maxDepth'])
            depth_user_normalized = depth_user_normalized *255 /depth_user_normalized.max()

            rows,cols = depth_user_normalized.shape
            M = numpy.float32([[1,0, shift[1]],[0,1, shift[0]]])
            affine_image = cv2.warpAffine(depth_user_normalized, M,(cols, rows))
            resize_image = imresize(affine_image, scale)
            resize_image_median = cv2.medianBlur(resize_image,5)

            rows, cols = resize_image_median.shape
            image_crop = resize_image_median[rows/2-160:rows/2+160, cols/2-160:cols/2+160]
            resize_final_out = imresize(image_crop, (IM_SZ,IM_SZ))
            if show_flag: # show the segmented images here
                cv2.imshow('image',image_crop)
                cv2.waitKey(10)

        return [resize_final_out, flag]





class ActionSample(object):
    """ Class that allows to access all the information for a certain action database sample """
    #define class to access actions data samples
    def __init__ (self,fileName):
        """ Constructor. Read the sample file and unzip it if it is necessary. All the data is loaded.

            sample=ActionSample('Sec01.zip')

        """
        # Check the given file
        if not os.path.exists(fileName) and not os.path.isfile(fileName):
            raise Exception("Sample path does not exist: " + fileName)

        # Prepare sample information
        self.fullFile = fileName
        self.dataPath = os.path.split(fileName)[0]
        self.file=os.path.split(fileName)[1]
        self.seqID=os.path.splitext(self.file)[0]
        self.samplePath=self.dataPath + os.path.sep + self.seqID;

        # Unzip sample if it is necessary
        if os.path.isdir(self.samplePath) :
            self.unzip = False
        else:
            self.unzip = True
            zipFile=zipfile.ZipFile(self.fullFile,"r")
            zipFile.extractall(self.samplePath)

        # Open video access for RGB information
        rgbVideoPath=self.samplePath + os.path.sep + self.seqID + '_color.mp4'
        if not os.path.exists(rgbVideoPath):
            raise Exception("Invalid sample file. RGB data is not available")
        self.rgb = cv2.VideoCapture(rgbVideoPath)
        while not self.rgb.isOpened():
            self.rgb = cv2.VideoCapture(rgbVideoPath)
            cv2.waitKey(500)

        # Read sample data
        sampleDataPath=self.samplePath + os.path.sep + self.seqID + '_data.csv'
        if not os.path.exists(sampleDataPath):
            raise Exception("Invalid sample file. Sample data is not available")
        self.data=dict()
        with open(sampleDataPath, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                self.data['numFrames']=int(row[0])
            del filereader

        # Read labels data
        labelsPath=self.samplePath + os.path.sep + self.seqID + '_labels.csv'
        self.labels=[]
        if not os.path.exists(labelsPath):
            warnings.warn("Labels are not available", Warning)
        else:
            with open(labelsPath, 'rb') as csvfile:
                filereader = csv.reader(csvfile, delimiter=',')
                for row in filereader:
                    self.labels.append(map(int,row))
                del filereader

    def __del__(self):
        """ Destructor. If the object unziped the sample, it remove the temporal data """
        if self.unzip:
            self.clean()

    def clean(self):
        """ Clean temporal unziped data """
        del self.rgb;
        shutil.rmtree(self.samplePath)

    def getFrame(self,video, frameNum):
        """ Get a single frame from given video object """
        # Check frame number
        # Get total number of frames
        numFrames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        # Check the given file
        if frameNum<1 or frameNum>numFrames:
            raise Exception("Invalid frame number <" + str(frameNum) + ">. Valid frames are values between 1 and " + str(int(numFrames)))
            # Set the frame index
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frameNum-1)
        ret,frame=video.read()
        if ret==False:
            raise Exception("Cannot read the frame")
        return frame

    def getNumFrames(self):
        """ Get the number of frames for this sample """
        return self.data['numFrames']


    def getRGB(self, frameNum):
        """ Get the RGB color image for the given frame """
        #get RGB frame
        return self.getFrame(self.rgb,frameNum)

    def getActions(self):
        """ Get the list of gesture for this sample. Each row is an action, with the format (actionID,startFrame,endFrame) """
        return self.labels

    def getActionsName(self,actionID):
        """ Get the action label from a given action ID """
        names=('wave','point','clap','crouch','jump','walk','run','shake hands', \
               'hug','kiss','fight')
        # Check the given file
        if actionID<1 or actionID>11:
            raise Exception("Invalid action ID <" + str(actionID) + ">. Valid IDs are values between 1 and 11")
        return names[actionID-1]
    def exportPredictions(self, prediction,predPath):
        """ Export the given prediction to the correct file in the given predictions path """
        if not os.path.exists(predPath):
            os.makedirs(predPath)
        output_filename = os.path.join(predPath,  self.seqID + '_prediction.csv')
        output_file = open(output_filename, 'wb')
        for row in prediction:
            output_file.write(repr(int(row[0])) + "," + repr(int(row[1])) + "," + repr(int(row[2])) + "\n")
        output_file.close()

    def evaluate(self,csvpathpred):
        """ Evaluate this sample agains the ground truth file """
        maxGestures=11
        seqLength=self.getNumFrames()

        # Get the list of gestures from the ground truth and frame activation
        predGestures = []
        binvec_pred = numpy.zeros((maxGestures, seqLength))
        gtGestures = []
        binvec_gt = numpy.zeros((maxGestures, seqLength))
        with open(csvpathpred, 'rb') as csvfilegt:
            csvgt = csv.reader(csvfilegt)
            for row in csvgt:
                binvec_pred[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
                predGestures.append(int(row[0]))

        # Get the list of gestures from prediction and frame activation
        for row in self.getActions():
                binvec_gt[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
                gtGestures.append(int(row[0]))

        # Get the list of gestures without repetitions for ground truth and predicton
        gtGestures = numpy.unique(gtGestures)
        predGestures = numpy.unique(predGestures)

        # Find false positives
        falsePos=numpy.setdiff1d(gtGestures, numpy.union1d(gtGestures,predGestures))

        # Get overlaps for each gesture
        overlaps = []
        for idx in gtGestures:
            intersec = sum(binvec_gt[idx-1] * binvec_pred[idx-1])
            aux = binvec_gt[idx-1] + binvec_pred[idx-1]
            union = sum(aux > 0)
            overlaps.append(intersec/union)

        # Use real gestures and false positive gestures to calculate the final score
        return sum(overlaps)/(len(overlaps)+len(falsePos))


class PoseSample(object):
    """ Class that allows to access all the information for a certain pose database sample """
    #define class to access gesture data samples
    def __init__ (self,fileName):

        """ Constructor. Read the sample file and unzip it if it is necessary. All the data is loaded.

            sample=PoseSample('Seq01.zip')

        """
        # Check the given file
        if not os.path.exists(fileName) and not os.path.isfile(fileName):
            raise Exception("Sequence path does not exist: " + fileName)

        # Prepare sample information
        self.fullFile = fileName
        self.dataPath = os.path.split(fileName)[0]
        self.file=os.path.split(fileName)[1]
        self.seqID=os.path.splitext(self.file)[0]
        self.samplePath=self.dataPath + os.path.sep + self.seqID;

        # Unzip sample if it is necessary
        if os.path.isdir(self.samplePath):
            self.unzip = False
        else:
            self.unzip = True
            zipFile=zipfile.ZipFile(self.fullFile,"r")
            zipFile.extractall(self.samplePath)

        # Set path for rgb images
        rgbPath=self.samplePath + os.path.sep + 'imagesjpg'+ os.path.sep
        if not os.path.exists(rgbPath):
            raise Exception("Invalid sample file. RGB data is not available")
        self.rgbpath = rgbPath


        # Set path for gt images
        gtPath=self.samplePath + os.path.sep + 'maskspng'+ os.path.sep
        if not os.path.exists(gtPath):
            self.gtpath= "empty"
        else:
            self.gtpath = gtPath


        frames=os.listdir(self.rgbpath)
        self.numberFrames=len(frames)


    def __del__(self):
        """ Destructor. If the object unziped the sample, it remove the temporal data """
        if self.unzip:
            self.clean()

    def clean(self):
        """ Clean temporal unziped data """
        shutil.rmtree(self.samplePath)



    def getRGB(self, frameNum):
        """ Get the RGB color image for the given frame """
        #get RGB frame
        if frameNum>self.numberFrames:
            raise Exception("Number of frame has to be less than: "+ self.numberFrames)
        framepath=self.rgbpath+self.seqID[3:5]+'_'+ '%04d' %frameNum+'.jpg'
        if not os.path.isfile(framepath):
            raise Exception("RGB file does not exist: " + framepath)
        return cv2.imread(framepath)

    def getNumFrames(self):
        return self.numberFrames

    def getLimb(self, frameNum, actorID,limbID):
        """ Get the BW limb image for a certain frame and a certain limbID """

        if self.gtpath == "empty":

            raise Exception("Limb labels are not available for this sequence. This sequence belong to the validation set.")
        else:

            limbpath=self.gtpath+self.seqID[3:5]+'_'+ '%04d' %frameNum+'_'+str(actorID)+'_'+str(limbID)+'.png'
            if frameNum>self.numberFrames:
                raise Exception("Number of frame has to be less than: "+ self.numberFrames)

            if actorID<1 or actorID>2:
                raise Exception("Invalid actor ID <" + str(actorID) + ">. Valid frames are values between 1 and 2 ")

            if limbID<1 or limbID>14:
                raise Exception("Invalid limb ID <" + str(limbID) + ">. Valid frames are values between 1 and 14")

            return cv2.imread(limbpath,cv2.CV_LOAD_IMAGE_GRAYSCALE)


    def getLimbsName(self,limbID):
        """ Get the limb label from a given limb ID """
        names=('head','torso','lhand','rhand','lforearm','rforearm','larm','rarm', \
               'lfoot','rfoot','lleg','rleg','lthigh','rthigh')
        # Check the given file
        if limbID<1 or limbID>14:
            raise Exception("Invalid limb ID <" + str(limbID) + ">. Valid IDs are values between 1 and 14")
        return names[limbID-1]

    def overlap_images(self, gtimage, predimage):

        """ this function computes the hit measure of overlap between two binary images im1 and im2 """


        [ret, im1] = cv2.threshold(gtimage,  127, 255, cv2.THRESH_BINARY)
        [ret, im2] = cv2.threshold(predimage, 127, 255, cv2.THRESH_BINARY)
        intersec = cv2.bitwise_and(im1, im2)
        intersec_val = float(numpy.sum(intersec))
        union = cv2.bitwise_or(im1, im2)
        union_val = float(numpy.sum(union))

        if union_val == 0:
            return 0
        else:
            if float(intersec_val / union_val)>0.5:
                return 1
            else:
                return 0

    def exportPredictions(self, prediction,frame,actor,limb,predPath):
        """ Export the given prediction to the correct file in the given predictions path """
        if not os.path.exists(predPath):
            os.makedirs(predPath)

        prediction_filename = predPath+os.path.sep+ self.seqID[3:5] +'_'+ '%04d' %frame +'_'+str(actor)+'_'+str(limb)+'_prediction.png'
        cv2.imwrite(prediction_filename,prediction)

    def evaluate(self, predpath):
        """ Evaluate this sample agains the ground truth file """
            # Get the list of videos from ground truth
        gt_list = os.listdir(self.gtpath)

        # For each sample on the GT, search the given prediction
        score = 0.0
        nevals = 0

        for gtlimbimage in gt_list:
            # Avoid double check, use only labels file
            if not gtlimbimage.lower().endswith(".png"):
                continue

            # Build paths for prediction and ground truth files
            aux = gtlimbimage.split('.')
            parts = aux[0].split('_')
            seqID = parts[0]
            gtlimbimagepath = os.path.join(self.gtpath,gtlimbimage)
            predlimbimagepath= os.path.join(predpath) + os.path.sep + seqID+'_'+parts[1]+'_'+parts[2]+'_'+parts[3]+"_prediction.png"

            #check predfile exists
            if not os.path.exists(predlimbimagepath) or not os.path.isfile(predlimbimagepath):
                raise Exception("Invalid video limb prediction file. Not all limb predictions are available")

            #Load images
            gtimage=cv2.imread(gtlimbimagepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            predimage=cv2.imread(predlimbimagepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)

            if cv2.cv.CountNonZero(cv2.cv.fromarray(gtimage)) >= 1:
                score += self.overlap_images(gtimage, predimage)
                nevals += 1

        #release videos and return mean overlap
        return score/nevals
