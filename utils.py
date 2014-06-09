""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""


import numpy


def  IsLeftDominant ( Skeleton_matrix ):
    """
    Check wether the motion is left dominant or right dominant
    """

    elbowDiffLeft = Skeleton_matrix[1:, 0:12] - Skeleton_matrix[0:-1, 0:12]
    elbowDiffRigh = Skeleton_matrix[1:, 12:24] - Skeleton_matrix[0:-1, 12:24]

    motionLeft = numpy.sum( numpy.sqrt( numpy.sum(elbowDiffLeft**2)))
    motionRigh = numpy.sum( numpy.sqrt( numpy.sum(elbowDiffRigh**2)))

    if motionLeft > motionRigh:
        leftDominantFlag = True
    else:
        leftDominantFlag = False
    return leftDominantFlag

def Extract_feature_UNnormalized(smp, used_joints, startFrame, endFrame):
    """
    Extract original features
    """
    frame_num = 0 
    Skeleton_matrix  = numpy.zeros(shape=(endFrame-startFrame+1, len(used_joints)*3))

    for numFrame in range(startFrame,endFrame+1):                    
        # Get the Skeleton object for this frame
        skel=smp.getSkeleton(numFrame)
        for joints in range(len(used_joints)):
            Skeleton_matrix[frame_num, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
        frame_num += 1

    
    if numpy.allclose(sum(sum(numpy.abs(Skeleton_matrix))),0):
        valid_skel = False
    else:
        valid_skel = True

    return Skeleton_matrix, valid_skel




def Extract_feature_normalized(smp, used_joints, startFrame, endFrame):
    """
    Extract normalized features
    """
    frame_num = 0 
    Skeleton_matrix  = numpy.zeros(shape=(endFrame-startFrame+1, len(used_joints)*3))
    normalized_joints = ['HipCenter', 'Spine', 'HipLeft', 'HipRight']
    HipCentre_matrix = numpy.zeros(shape=(endFrame-startFrame+1, len(normalized_joints)*3))

    for numFrame in range(startFrame,endFrame+1):                    
        # Get the Skeleton object for this frame
        skel=smp.getSkeleton(numFrame)
        for joints in range(len(used_joints)):
            Skeleton_matrix[frame_num, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
        for joints in range(len(normalized_joints)):
            HipCentre_matrix[frame_num, joints*3: (joints+1)*3] = skel.joins[normalized_joints[joints]][0]             

        frame_num += 1

    xCentLst = HipCentre_matrix[:, range(0,10,3)]
    xCentVal = sum(sum(xCentLst)) / (xCentLst.shape[0]*xCentLst.shape[1])

    yCentLst = HipCentre_matrix[:, range(1,11,3)]
    yCentVal = sum(sum(yCentLst)) / (yCentLst.shape[0]*yCentLst.shape[1])

    zCentLst = HipCentre_matrix[:, range(2,12,3)]
    zCentVal = sum(sum(zCentLst)) / (zCentLst.shape[0]*zCentLst.shape[1])

    Skeleton_matrix[:, range(0,10,3)] = Skeleton_matrix[:, range(0,10,3)] - xCentVal
    Skeleton_matrix[:, range(1,11,3)] = Skeleton_matrix[:, range(1,11,3)] - yCentVal
    Skeleton_matrix[:, range(2,12,3)] = Skeleton_matrix[:, range(2,12,3)] - zCentVal

    xCentLst -= xCentVal
    yCentLst -= yCentVal
    zCentLst -= zCentVal

    coordHip = [xCentLst[:,0], yCentLst[:,0], zCentLst[:,0]]
    coordHip = numpy.mean(coordHip, axis=1)

    coordShou = [xCentLst[:,1], yCentLst[:,1], zCentLst[:,1]]
    coordShou = numpy.mean(coordShou, axis=1)

    scaleRatio = (sum(coordHip - coordShou)**2)**0.5

    Skeleton_matrix = Skeleton_matrix / scaleRatio

    if scaleRatio==0:
        valid_skel = False
    else:
        valid_skel = True
    return Skeleton_matrix, valid_skel

def Extract_feature_normalized_ALL(smp, used_joints, startFrame, endFrame):
    """
    Extract normalized features, but we replicate the first undetected frames as the 
    last detected frames
    """
    frame_num = 0 
    Skeleton_matrix  = numpy.zeros(shape=(endFrame-startFrame+1, len(used_joints)*3))
    normalized_joints = ['HipCenter', 'Spine', 'HipLeft', 'HipRight']
    HipCentre_matrix = numpy.zeros(shape=(endFrame-startFrame+1, len(normalized_joints)*3))


    Start_frame = 0
    ### first detect initial frames are valid:
    for numFrame in range(startFrame,endFrame):                    
    # Get the Skeleton object for this frame
        skel=smp.getSkeleton(numFrame)
        for joints in range(len(used_joints)):
            Skeleton_matrix[frame_num, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
        if sum(Skeleton_matrix[frame_num, :])==0:
            Start_frame = numFrame
            skel=smp.getSkeleton(numFrame+1)
            Skeleton_matrix[frame_num, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
            if sum(Skeleton_matrix[frame_num, :])!=0:
                break

    Take_Frame = endFrame
    while(1):
        skel=smp.getSkeleton(Take_Frame)
        Skeleton_matrix_temp  = numpy.zeros(shape=(1, len(used_joints)*3))
        for joints in range(len(used_joints)):
            Skeleton_matrix_temp[:, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
        if sum(sum(Skeleton_matrix_temp))!=0:
                break
        else:
            Take_Frame -= 1
            print "missing frame"+str(Take_Frame)


    for numFrame in range(0,Start_frame):                    
        # Get the Skeleton object for this frame
        skel=smp.getSkeleton(Take_Frame)
        for joints in range(len(used_joints)):
            Skeleton_matrix[numFrame, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
        for joints in range(len(normalized_joints)):
            HipCentre_matrix[numFrame, joints*3: (joints+1)*3] = skel.joins[normalized_joints[joints]][0]  



    for numFrame in range(Start_frame,endFrame):                    
        # Get the Skeleton object for this frame
        skel=smp.getSkeleton(numFrame+1)
        for joints in range(len(used_joints)):
            Skeleton_matrix[numFrame, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
        for joints in range(len(normalized_joints)):
            HipCentre_matrix[numFrame, joints*3: (joints+1)*3] = skel.joins[normalized_joints[joints]][0]             

    xCentLst = HipCentre_matrix[:, range(0,10,3)]
    xCentVal = sum(sum(xCentLst)) / (xCentLst.shape[0]*xCentLst.shape[1])

    yCentLst = HipCentre_matrix[:, range(1,11,3)]
    yCentVal = sum(sum(yCentLst)) / (yCentLst.shape[0]*yCentLst.shape[1])

    zCentLst = HipCentre_matrix[:, range(2,12,3)]
    zCentVal = sum(sum(zCentLst)) / (zCentLst.shape[0]*zCentLst.shape[1])

    Skeleton_matrix[:, range(0,10,3)] = Skeleton_matrix[:, range(0,10,3)] - xCentVal
    Skeleton_matrix[:, range(1,11,3)] = Skeleton_matrix[:, range(1,11,3)] - yCentVal
    Skeleton_matrix[:, range(2,12,3)] = Skeleton_matrix[:, range(2,12,3)] - zCentVal

    xCentLst -= xCentVal
    yCentLst -= yCentVal
    zCentLst -= zCentVal

    coordHip = [xCentLst[:,0], yCentLst[:,0], zCentLst[:,0]]
    coordHip = numpy.mean(coordHip, axis=1)

    coordShou = [xCentLst[:,1], yCentLst[:,1], zCentLst[:,1]]
    coordShou = numpy.mean(coordShou, axis=1)

    scaleRatio = (sum(coordHip - coordShou)**2)**0.5

    Skeleton_matrix = Skeleton_matrix / scaleRatio

    if scaleRatio==0:
        valid_skel = False
    else:
        valid_skel = True
    return Skeleton_matrix, valid_skel


def Extract_feature(Pose, njoints):
    #Fcc
    FeatureNum = 0
    Fcc =  numpy.zeros(shape=(Pose.shape[0], njoints * (njoints-1)/2*3))
    for joints1 in range(njoints-1):
        for joints2 in range(joints1+1,njoints):
            Fcc[:, FeatureNum*3:(FeatureNum+1)*3] = Pose[:, joints1*3:(joints1+1)*3]-Pose[:, joints2*3:(joints2+1)*3];
            FeatureNum += 1
            
    #F_cp
    FeatureNum = 0
    Fcp = numpy.zeros(shape=(Pose.shape[0]-1, njoints **2*3))
    for joints1 in range(njoints):
        for joints2 in range(njoints):
            Fcp[:, FeatureNum*3: (FeatureNum+1)*3] = Pose[1:,joints1*3:(joints1+1)*3]-Pose[0:-1,joints2*3:(joints2+1)*3]
            FeatureNum += 1
              
    #Instead of initial frame as in the paper Eigenjoints-based action recognition using
    #naive-bayes-nearest-neighbor, we use final frame because it's better initiated
    # F_cf
    FeatureNum = 0
    Pose_final = numpy.tile(Pose [-1 , :] , [Pose.shape[0], 1])
    Fcf = numpy.zeros(shape=(Pose.shape[0]-1, njoints **2*3))
    for joints1 in range(njoints):
        for joints2 in range(njoints):
                Fcf[:, FeatureNum*3: (FeatureNum+1)*3] = Pose[0:-1, joints1*3:(joints1+1)*3] - Pose_final[0:-1,joints2*3:(joints2+1)*3]
                FeatureNum=FeatureNum+1

    Features = numpy.concatenate( (Fcc[0:-1, :], Fcp, Fcf), axis = 1)
    return Features

def Extract_feature_Realtime(Pose, njoints):
    #Fcc
    FeatureNum = 0
    Fcc =  numpy.zeros(shape=(Pose.shape[0], njoints * (njoints-1)/2*3))
    for joints1 in range(njoints-1):
        for joints2 in range(joints1+1,njoints):
            Fcc[:, FeatureNum*3:(FeatureNum+1)*3] = Pose[:, joints1*3:(joints1+1)*3]-Pose[:, joints2*3:(joints2+1)*3];
            FeatureNum += 1
            
    #F_cp
    FeatureNum = 0
    Fcp = numpy.zeros(shape=(Pose.shape[0]-1, njoints **2*3))
    for joints1 in range(njoints):
        for joints2 in range(njoints):
            Fcp[:, FeatureNum*3: (FeatureNum+1)*3] = Pose[1:,joints1*3:(joints1+1)*3]-Pose[0:-1,joints2*3:(joints2+1)*3]
            FeatureNum += 1
              
    #Instead of initial frame as in the paper Eigenjoints-based action recognition using
    #naive-bayes-nearest-neighbor, we use final frame because it's better initiated
    # F_cf

    Features = numpy.concatenate( (Fcc[0:-1, :], Fcp), axis = 1)
    return Features

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def zero_mean_unit_variance(Data):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    Mean = numpy.mean(Data, axis=0)
    Data  -=  Mean

    Std = numpy.std(Data, axis = 0)
    index = (numpy.abs(Std<10**-5))
    Std[index] = 1
    Data /= Std
    return [Data, Mean, Std]


def normalize(Data, Mean, Std):
    Data -= Mean
    Data /= Std
    return Data


def plot_skeleton_energy(smp):
    Energy = numpy.zeros(shape=(smp.getNumFrames()-1,1))
    Skeleton_1 = numpy.zeros(shape=(1, len(used_joints)*3))
    Skeleton_2 = numpy.zeros(shape=(1, len(used_joints)*3))
    for frame_num in range(1,smp.getNumFrames()):
                # Get the Skeleton object for this frame
        skel_1=smp.getSkeleton(frame_num)
        skel_2=smp.getSkeleton(frame_num+1)
        for joints in range(len(used_joints)):
            Skeleton_1[0, joints*3: (joints+1)*3] = skel_1.joins[used_joints[joints]][0]
            Skeleton_2[0, joints*3: (joints+1)*3] = skel_2.joins[used_joints[joints]][0]

        Energy[frame_num-1] = sum(sum((Skeleton_1 - Skeleton_2)**2))

            

    from scipy.signal import lfilter

    windowSize = 20
    frmPwrList = lfilter(numpy.ones(windowSize) / windowSize, 1, Energy)
    plt.figure()
    ax = plt.gca()
    plt.plot(out)
    plt.draw()

    for gesture in gesturesList:
        # Get the gesture ID, and start and end frames for the gesture
        ax = plt.gca()
        gestureID,startFrame,endFrame=gesture
        #r = matplotlib.patches.Rectangle((startFrame, 0),  endFrame,0.003, fill=False)
        vlines(startFrame, 0, 0.003, colors='k', linestyles='solid')
        vlines(endFrame, 0, 0.003, colors='r', linestyles='solid')
        print "beginL %d, end: %d"%(startFrame,endFrame)



def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array


def viterbi_path(prior, transmat, observ_likelihood):
    """ Viterbi path decoding
    Wudi first implement the forward pass.
    Future works include forward-backward encoding
    input: prior probability 1*N...
    transmat: N*N
    observ_likelihood: N*T
    """
    T = observ_likelihood.shape[-1]
    N = observ_likelihood.shape[0]

    path = numpy.zeros(T, dtype=numpy.int32)
    global_score = numpy.zeros(shape=(N,T))
    predecessor_state_index = numpy.zeros(shape=(N,T), dtype=numpy.int32)

    t = 1
    global_score[:, 0] = prior * observ_likelihood[:, 0]
    # need to  normalize the data
    global_score[:, 0] = global_score[:, 0] /sum(global_score[:, 0] )
    
    for t in range(1, T):
        for j in range(N):
            temp = global_score[:, t-1] * transmat[:, j] * observ_likelihood[j, t]
            global_score[j, t] = max(temp)
            predecessor_state_index[j, t] = temp.argmax()

        global_score[:, t] = global_score[:, t] / sum(global_score[:, t])

    path[T-1] = global_score[:, T-1].argmax()
    
    for t in range(T-2, -1, -1):
        path[t] = predecessor_state_index[ path[t+1], t+1]

    return [path, predecessor_state_index, global_score]


def viterbi_path_log(prior, transmat, observ_likelihood):
    """ Viterbi path decoding
    Wudi first implement the forward pass.
    Future works include forward-backward encoding
    input: prior probability 1*N...
    transmat: N*N
    observ_likelihood: N*T
    """
    T = observ_likelihood.shape[-1]
    N = observ_likelihood.shape[0]

    path = numpy.zeros(T, dtype=numpy.int32)
    global_score = numpy.zeros(shape=(N,T))
    predecessor_state_index = numpy.zeros(shape=(N,T), dtype=numpy.int32)

    t = 1
    global_score[:, 0] = prior + observ_likelihood[:, 0]
    # need to  normalize the data
    
    for t in range(1, T):
        for j in range(N):
            temp = global_score[:, t-1] + transmat[:, j] + observ_likelihood[j, t]
            global_score[j, t] = max(temp)
            predecessor_state_index[j, t] = temp.argmax()

    path[T-1] = global_score[:, T-1].argmax()
    
    for t in range(T-2, -1, -1):
        path[t] = predecessor_state_index[ path[t+1], t+1]

    return [path, predecessor_state_index, global_score]


def imdisplay(im):
    """ display grayscale images
    """
    im_min = im.min()
    im_max = im.max()
    return (im - im_min) / (im_max -im_min)


def viterbi_colab_clean(path, global_score, threshold=-3, mini_frame=15):
    """
    Clean the viterbi path output according to its global score,
    because some are out of the vocabulary
    """

# just to accommodate some frame didn't start right from the begining
    start_label = numpy.concatenate((range(0,200,10), range(1,200,10), range(2,200,10), range(3,200,10) ))
    end_label   = numpy.concatenate((range(9,200,10), range(8,200,10), range(7,200,10), range(6,200,10) ) )
    begin_frame = []
    end_frame = []
    pred_label = []

    frame = 1
    while(frame< path.shape[-1]-1):
        if path[frame-1]==200 and path[frame] in start_label:
            begin_frame.append(frame)
            # python integer divsion will do the floor for us :)
            pred_label .append( path[frame]/10 + 1)
            frame += 1
        elif path[frame] in end_label and path[frame+1]==200:
            end_frame.append(frame)
            frame += 1
        frame += 1
    

    end_frame = numpy.array(end_frame)
    begin_frame = numpy.array(begin_frame)
    pred_label= numpy.array(pred_label)
    # risky hack! just for validation file 663
    if len(begin_frame)> len(end_frame):
        begin_frame = begin_frame[:-1]
    elif len(begin_frame)< len(end_frame):# risky hack! just for validation file 668
        end_frame = end_frame[1:]
    ## First delete the predicted gesture less than 15 frames
    frame_length = end_frame - begin_frame
    ## now we delete the gesture outside the vocabulary by choosing
    ## frame number small than mini_frame
    mask = frame_length > mini_frame
    begin_frame = begin_frame[mask]
    end_frame = end_frame[mask]
    pred_label = pred_label[mask]


    Individual_score = []
    for idx, g in enumerate(begin_frame):
            score_start = global_score[path[g], g]
            score_end = global_score[path[end_frame[idx]], end_frame[idx]]
            Individual_score.append(score_end - score_start)
    ## now we delete the gesture outside the vocabulary by choosing
    ## score lower than a threshold
    Individual_score = numpy.array(Individual_score)
    frame_length = end_frame - begin_frame
    # should be length independent
    Individual_score = Individual_score/frame_length

    mask = Individual_score > threshold
    begin_frame = begin_frame[mask]
    end_frame = end_frame[mask]
    pred_label = pred_label[mask]
    Individual_score = Individual_score[mask]
    

    return [pred_label, begin_frame, end_frame, Individual_score, frame_length]


def createSubmisionFile(predictionsPath,submisionPath):
    """ Create the submission file, ready to be submited to Codalab. """
    import os, zipfile
    # Create the output path and remove any old file
    if os.path.exists(submisionPath):
        oldFileList = os.listdir(submisionPath);
        for file in oldFileList:
            os.remove(os.path.join(submisionPath,file));
    else:
        os.makedirs(submisionPath);

    # Create a ZIP with all files in the predictions path
    zipf = zipfile.ZipFile(os.path.join(submisionPath,'Submission.zip'), 'w');
    for root, dirs, files in os.walk(predictionsPath):
        for file in files:
            zipf.write(os.path.join(root, file), file, zipfile.ZIP_DEFLATED);
    zipf.close()