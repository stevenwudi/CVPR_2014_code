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
from numpy import log
from shutil import copyfile
import matplotlib.pyplot as plt
import cv2
from ChalearnLAPEvaluation import evalGesture,exportGT_Gesture
from ChalearnLAPSample import GestureSample
from utils import IsLeftDominant
from utils import Extract_feature_Realtime
from utils import Extract_feature_UNnormalized
from utils import normalize
from utils import imdisplay
from utils import viterbi_colab_clean
from utils import createSubmisionFile
import time
import cPickle
import numpy
import pickle
import scipy.io as sio  

### theano import
sys.path.append(r'C:\Users\PC-User\Documents\Visual Studio 2012\Projects\Theano\Tutorial')
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from rbm import RBM
from grbm import GBRBM
from utils import zero_mean_unit_variance
from utils import normalize
from GRBM_DBN import GRBM_DBN
from sklearn import preprocessing
############### viterbi path import
from utils import viterbi_path, viterbi_path_log

#########################

""" Main script. Show how to perform all competition steps
    Access the sample information to learn a model. """
# Data folder (Training data)
print("Extracting the training files")
data_path=os.path.join("I:\Kaggle_multimodal\Training\\")  
# Predictions folder (output)
outPred='./training/pred/'
# Get the list of training samples
samples=os.listdir(data_path)
used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft','HandLeft',
                'ElbowRight', 'WristRight','ShoulderRight','HandRight',
                'Head','Spine','HipCenter']
njoints = len(used_joints)
STATE_NO = 10
count = 0

### load the pre-store normalization constant
f = open('SK_normalization.pkl','rb')
SK_normalization = cPickle.load(f)
Mean1 = SK_normalization ['Mean1']
Std1 = SK_normalization['Std1']

## Load Prior and transitional Matrix
dic=sio.loadmat('Transition_matrix.mat')
Transition_matrix = dic['Transition_matrix']
Prior = dic['Prior']

for file_count, file in enumerate(samples):
    #if not file.endswith(".zip"):
    #    continue;  
    time_tic = time.time()      
    if  not file_count<650:
        print("\t Processing file " + file)
        # Create the object to access the sample
        smp=GestureSample(os.path.join(data_path,file))
        # ###############################################
        # USE Ground Truth information to learn the model
        # ###############################################
        # Get the list of actions for this frame
        gesturesList=smp.getGestures()
        ###########################################################
        # we check whether it's left dominant or right dominanant
        # if right dominant, we correct them to left dominant
        ##########################################################
        Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, used_joints, 1, smp.getNumFrames())

        Feature = Extract_feature_Realtime(Skeleton_matrix, njoints)

        Feature_normalized = normalize(Feature, Mean1, Std1)

        ### Feed into DBN
        shared_x = theano.shared(numpy.asarray(Feature_normalized,
                                    dtype=theano.config.floatX),
                                    borrow=True)
        numpy_rng = numpy.random.RandomState(123)

        ### model 1
        ##########################
        dbn = GRBM_DBN(numpy_rng=numpy_rng, n_ins=528,
        hidden_layers_sizes=[1000, 1000, 500],
        n_outs=201)
        dbn.load('dbn_2014-05-23-20-07-28.npy')

        validate_model = theano.function(inputs=[],
            outputs=dbn.logLayer.p_y_given_x,
            givens={ dbn.x: shared_x})

        observ_likelihood_1 = validate_model()
        del dbn
        ### model 2
        ##########################
        dbn = GRBM_DBN(numpy_rng=numpy_rng, n_ins=528,
        hidden_layers_sizes=[1000, 1000, 500],
        n_outs=201)

        dbn.load('dbn_2014-05-24-05-53-17.npy')

        validate_model = theano.function(inputs=[],
            outputs=dbn.logLayer.p_y_given_x,
            givens={ dbn.x: shared_x})

        observ_likelihood_2 = validate_model()
        del dbn
        ### model 3
        ##########################
        
        dbn = GRBM_DBN(numpy_rng=numpy_rng, n_ins=528,
            hidden_layers_sizes=[2000, 2000, 1000],
            n_outs=201, finetune_lr=0.1)
        dbn.load('dbn_2014-05-25-10-11-56.npy')
        # Optimization complete with best validation score of 38.194915 %,with test performance 38.113636 %
        #....The score for this prediction is 0.792685963841

        validate_model = theano.function(inputs=[],
            outputs=dbn.logLayer.p_y_given_x,
            givens={ dbn.x: shared_x})

        observ_likelihood_3 = validate_model()
        del dbn

        ### model 4
        ##########################
        dbn = GRBM_DBN(numpy_rng=numpy_rng, n_ins=528,
            hidden_layers_sizes=[2000, 2000, 1000],
            n_outs=201, finetune_lr=0.1)
        dbn.load('dbn_2014-05-25-10-11-56.npy')
        # Optimization complete with best validation score of 38.194915 %,with test performance 38.113636 %
        #The score for this prediction is 0.777992357011

        validate_model = theano.function(inputs=[],
            outputs=dbn.logLayer.p_y_given_x,
            givens={ dbn.x: shared_x})

        observ_likelihood_4 = validate_model()
        del dbn
        #sio.savemat('observ_likelihood.mat', {'observ_likelihood':observ_likelihood})
        ##########################
        # viterbi path decoding
        #####################

        log_observ_likelihood = log(observ_likelihood_1.T) + log(observ_likelihood_2.T) \
                                 + log(observ_likelihood_3.T) #+ log(observ_likelihood_4.T)
        log_observ_likelihood[-1, 0:5] = 0
        log_observ_likelihood[-1, -5:] = 0

        print("\t Viterbi path decoding " )
        # do it in log space avoid numeric underflow
        [path, predecessor_state_index, global_score] = viterbi_path_log(log(Prior), log(Transition_matrix), log_observ_likelihood)
        #[path, predecessor_state_index, global_score] =  viterbi_path(Prior, Transition_matrix, observ_likelihood)
        
        # Some gestures are not within the vocabulary
        [pred_label, begin_frame, end_frame, Individual_score, frame_length] = viterbi_colab_clean(path, global_score, threshold=-100, mini_frame=19)


        ### In theory we need add frame, but it seems that the groutnd truth is about 3 frames more, a bit random
        end_frame = end_frame + 3
        ### plot the path and prediction
        if True:
            im  = imdisplay(global_score)
            plt.clf()
            plt.imshow(im, cmap='gray')
            plt.plot(range(global_score.shape[-1]), path, color='c',linewidth=2.0)
            plt.xlim((0, global_score.shape[-1]))
            # plot ground truth
            for gesture in gesturesList:
            # Get the gesture ID, and start and end frames for the gesture
                gestureID,startFrame,endFrame=gesture
                frames_count = numpy.array(range(startFrame, endFrame+1))
                pred_label_temp = ((gestureID-1) *10 +5) * numpy.ones(len(frames_count))
                plt.plot(frames_count, pred_label_temp, color='r', linewidth=5.0)
            
            # plot clean path
            for i in range(len(begin_frame)):
                frames_count = numpy.array(range(begin_frame[i], end_frame[i]+1))
                pred_label_temp = ((pred_label[i]-1) *10 +5) * numpy.ones(len(frames_count))
                plt.plot(frames_count, pred_label_temp, color='#ffff00', linewidth=2.0)

            plt.show()

            from pylab import savefig
            save_dir=r'.\training\SK_path'
            save_path= os.path.join(save_dir,file)
            savefig(save_path, bbox_inches='tight')
            plt.show()

        print "Elapsed time %d sec" % int(time.time() - time_tic)
        save_dir=r'.\training\SK'
        save_path= os.path.join(save_dir,file)
        out_file = open(save_path, 'wb')
        cPickle.dump({'log_observ_likelihood':log_observ_likelihood}, out_file, protocol=cPickle.HIGHEST_PROTOCOL)
        out_file.close()

        pred=[]
        for i in range(len(begin_frame)):
            pred.append([ pred_label[i], begin_frame[i], end_frame[i]] )

        smp.exportPredictions(pred,outPred)
     # ###############################################
        ## delete the sample
        del smp        



TruthDir='./training/gt/'
final_score = evalGesture(outPred,TruthDir)         
print("The score for this prediction is " + "{:.12f}".format(final_score))
#The score for this prediction is 0.746762613292  -inf threshold, dbn_2014-05-23-20-07-28
#The score for this prediction is 0.731507614243  -3 threshold, dbn_2014-05-23-20-07-28
#The score for this prediction is 0.748537955342 -inf threshold, dbn_2014-05-24-05-53-17
# Submision folder (output)
outSubmision='./training/submision/'
# Prepare submision file (only for validation and final evaluation data sets)
createSubmisionFile(outPred, outSubmision)




#Sample ID: Sample0651, score 0.857417
#Sample ID: Sample0652, score 0.913935
#Sample ID: Sample0653, score 0.943355
#Sample ID: Sample0654, score 0.917020
#Sample ID: Sample0655, score 0.924133
#Sample ID: Sample0656, score 0.749035
#Sample ID: Sample0657, score 0.955422
#Sample ID: Sample0658, score 0.908295
#Sample ID: Sample0659, score 0.859846
#Sample ID: Sample0660, score 0.856747
#Sample ID: Sample0661, score 0.914236
#Sample ID: Sample0662, score 0.786864
#Sample ID: Sample0663, score 0.941406
#Sample ID: Sample0664, score 0.828827
#Sample ID: Sample0665, score 0.853589
#Sample ID: Sample0666, score 0.720335
#Sample ID: Sample0667, score 0.853116
#Sample ID: Sample0668, score 0.933476
#Sample ID: Sample0669, score 0.881736
#Sample ID: Sample0670, score 0.768433
#Sample ID: Sample0671, score 0.909118
#Sample ID: Sample0672, score 0.950289
#Sample ID: Sample0673, score 0.744832
#Sample ID: Sample0674, score 0.859022
#Sample ID: Sample0675, score 0.092073
#Sample ID: Sample0676, score 0.954039
#Sample ID: Sample0677, score 0.794421
#Sample ID: Sample0678, score 0.709793
#Sample ID: Sample0679, score 0.809159
#Sample ID: Sample0680, score 0.812236
#Sample ID: Sample0681, score 0.684452
#Sample ID: Sample0682, score 0.828362
#Sample ID: Sample0683, score 0.650288
#Sample ID: Sample0684, score 0.820198
#Sample ID: Sample0685, score 0.948309
#Sample ID: Sample0686, score 0.774727
#Sample ID: Sample0687, score 0.870839
#Sample ID: Sample0688, score 0.770792
#Sample ID: Sample0689, score 0.957459
#Sample ID: Sample0690, score 0.929372
#Sample ID: Sample0691, score 0.912913
#Sample ID: Sample0692, score 0.921437
#Sample ID: Sample0693, score 0.917426
#Sample ID: Sample0694, score 0.819258
#Sample ID: Sample0695, score 0.930915
#Sample ID: Sample0696, score 0.913294
#Sample ID: Sample0697, score 0.904583
#Sample ID: Sample0698, score 0.807169
#Sample ID: Sample0699, score 0.000000
#Sample ID: Sample0700, score 0.814740
#The score for this prediction is 0.823574769630


####################True validation##############
#Sample ID: Sample0651, score 0.904025
#Sample ID: Sample0652, score 0.904918
#Sample ID: Sample0653, score 0.958563
#Sample ID: Sample0654, score 0.908786
#Sample ID: Sample0655, score 0.939971
#Sample ID: Sample0656, score 0.855592
#Sample ID: Sample0657, score 0.949432
#Sample ID: Sample0658, score 0.892104
#Sample ID: Sample0659, score 0.821147
#Sample ID: Sample0660, score 0.731472
#Sample ID: Sample0661, score 0.937361
#Sample ID: Sample0662, score 0.669438
#Sample ID: Sample0663, score 0.951005
#Sample ID: Sample0664, score 0.943669
#Sample ID: Sample0665, score 0.733362
#Sample ID: Sample0666, score 0.609271
#Sample ID: Sample0667, score 0.860603
#Sample ID: Sample0668, score 0.858290
#Sample ID: Sample0669, score 0.929701
#Sample ID: Sample0670, score 0.768116
#Sample ID: Sample0671, score 0.814299
#Sample ID: Sample0672, score 0.930511
#Sample ID: Sample0673, score 0.673121
#Sample ID: Sample0674, score 0.812634
#Sample ID: Sample0675, score 0.095109
#Sample ID: Sample0676, score 0.849760
#Sample ID: Sample0677, score 0.855732
#Sample ID: Sample0678, score 0.697313
#Sample ID: Sample0679, score 0.868751
#Sample ID: Sample0680, score 0.784426
#Sample ID: Sample0681, score 0.667418
#Sample ID: Sample0682, score 0.789869
#Sample ID: Sample0683, score 0.712648
#Sample ID: Sample0684, score 0.774973
#Sample ID: Sample0685, score 0.696109
#Sample ID: Sample0686, score 0.718954
#Sample ID: Sample0687, score 0.614459
#Sample ID: Sample0688, score 0.823834
#Sample ID: Sample0689, score 0.891862
#Sample ID: Sample0690, score 0.868217
#Sample ID: Sample0691, score 0.895659
#Sample ID: Sample0692, score 0.763341
#Sample ID: Sample0693, score 0.919345
#Sample ID: Sample0694, score 0.884368
#Sample ID: Sample0695, score 0.786327
#Sample ID: Sample0696, score 0.855285
#Sample ID: Sample0697, score 0.909057
#Sample ID: Sample0698, score 0.714707
#Sample ID: Sample0699, score 0.493874
#Sample ID: Sample0700, score 0.797374
#The score for this prediction is 0.801723171675