from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import  linalg as LA
import networkx as nx
from scipy import linalg as spLA

####################################
#### Preprocessing code ##############   
#######################################
def resize_images(imgs,img_rows, img_cols, interp_method='INTER_AREA'):
    interp_dict ={'INTER_AREA':cv2.INTER_AREA,
                'INTER_CUBIC': cv2.INTER_CUBIC,
                'INTER_LINEAR':cv2.INTER_LINEAR}

    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), 
            interpolation=interp_dict[interp_method])
    return imgs_p
            

def mean_center_and_scale(img_arr, scale=None, squeeze= True):
    """
    Input
    img_arr is (dataSetSize, pixelRows, PixelColumns) ndarray
    scale is global divisor of pixel intensity after mean centering
            if scale=None divide by max(abs())) pixel intensity after mean centering
            
    squeeze = If True then squeeze 
            all lenght 1 axes
    
    return img_arr after mean centering, scaling and possibly squeezing
    
    """
    if squeeze:
        img_arr =np.squeeze(img_arr)
        
    pixel_means =  np.mean(img_arr, axis=0)
    img_arr =  img_arr - pixel_means
    
    if scale is None:
        scale = np.max(np.abs(img_arr))
    img_arr = img_arr / np.float(scale)  
    return img_arr
    
def mask_not_blank(mask):
    return sum(mask.flatten()) > 0.1
    
###########################################
#### training/ValidationPartitioning #########
############################################

#def selectTrainIndices(pos_maks, neg_maks, trainSetSize)

def TrainingValditionIndexPartition(PosIndices, NegIndices, trainSetSize,
                        PosElemsLimiting=True):                   
    """
    PosIndices - 1d array indexing postive elements of numpy array version of
                kaggle data set
    NegIndices  - "".. negative elements... ""
   
   trainSetSize - integer size for the training set to be constructed
   PosElemsLimiting - Bool (True if the dataset to be partitioned has more negative
                        than positive elements)
   
    """  
    ## selecting indices for Training set                                                     
    TrainPosIndices = np.random.choice(PosIndices, size=np.int(trainSetSize/2), replace=False)
    TrainNegIndices = np.random.choice(NegIndices, size=np.int(trainSetSize/2), replace=False)

    print("shape of TrainPosIndices {}".format(np.shape(TrainPosIndices)) )
    print("shape of TrainNegIndices {}".format(np.shape(TrainNegIndices)) )
   
    # select indices for validation set
    if PosElemsLimiting:
        ValidPosIndices = np.setdiff1d(PosIndices, TrainPosIndices)
        ValidNegIndices = np.random.choice(  np.setdiff1d(NegIndices, TrainNegIndices), 
                                   size = len(ValidPosIndices), replace = False)   
    else:
        ValidNegIndices = np.setdiff1d(NegIndices, TrainNegIndices)
        ValidPosIndices = np.random.choice(  np.setdiff1d(PosIndices, TrainPosIndices), 
                                   size = len(ValidNegIndices), replace = False)  
    
    print("shape of ValidPosIndices {}".format(np.shape(ValidPosIndices)) )
    print("shape of ValidNegIndices {}".format(np.shape(ValidNegIndices)) )

    return [TrainPosIndices, TrainNegIndices], [ValidPosIndices, ValidNegIndices]

def makeTrainValidationSetsFromIndices(imgs , masks , trainSetIndices, validSetIndices):
    trainSet  = imgs[trainSetIndices]
    trainMasks = masks[trainSetIndices]
    
    validSet= imgs[validSetIndices]
    validMasks  = masks[validSetIndices]
    
    return [trainSet, trainMasks], [validSet, validMasks]
######################################                
# PCA code ###########################
######################################

def get_var_PC_vecs_LA(data):
    """
    data is (num_observations, num_features) matrix
    rows index observations column index indexes features
    """
    feature_means = np.mean(data,axis=0)
    mean_centered_data  = data - feature_means
    cov_mat =np.cov(mean_centered_data, rowvar=False)
    evals, evecs = LA.eigh(cov_mat)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    return evals, evecs, feature_means     
    
def get_var_PC_vecs_spEigh(data, nb_largest_ev=10):
    """
    data is (num_observations, num_features) matrix
    rows index observations column index indexes features
    """
    feature_means = np.mean(data,axis=0)
    mean_centered_data  = data - feature_means
    cov_mat =np.cov(mean_centered_data, rowvar=False)
    evals, evecs = spLA.eigh(cov_mat,
                    eigvals=(np.shape(cov_mat)[0] - nb_largest_ev ,np.shape(cov_mat)[0]-1)
                            )
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    return evals, evecs, feature_means      
     
def local_PCA(imgs, window=(10,10),stride=(1,1), nb_largest_ev=10): 
    """
    imgs (nb_imgs , img_rows (height) ,img_cols (width) ) array
    """
    assert len(np.shape(imgs))==3

    img_rows , img_cols =np.shape(imgs)[-2:]
    print("img_rows {}".format(img_rows))
    print("img_cols {}".format(img_cols))
    output_arr_dims =(
                    np.int(np.ceil((img_rows - window[0] + 1.0)/stride[0])), 
                    np.int(np.ceil((img_cols - window[1]  + 1.0)/stride[1]))  
                            ) 
                            
    print("output_arr_dims {}".format(output_arr_dims))
    nb_evals = np.int( min(np.prod(window) , nb_largest_ev )  )
    
    local_PCA_evals = np.zeros(
                        shape=(output_arr_dims[0], output_arr_dims[1],nb_evals ),
                        dtype=np.float)
    local_PCA_evecs =np.zeros(
                            shape=(output_arr_dims[0], output_arr_dims[1],np.prod(window),nb_evals),
                            dtype=np.float
                            )
    
    for row_index  in xrange(0,img_rows - window[0] + 1, stride[0]):
        for col_index  in xrange(0,img_cols - window[1] + 1, stride[1]):
        
            local_imgs =imgs[:,row_index:row_index + window[0], col_index:col_index + window[1]]
            local_imgs = np.reshape(local_imgs, 
                                newshape=(np.shape(local_imgs)[0],np.prod(window))
                                                )
            local_evals, local_evecs, _ = get_var_PC_vecs_spEigh(local_imgs, nb_largest_ev )
            local_PCA_evals[row_index//stride[0],col_index//stride[1],:] =  local_evals                                     
            local_PCA_evecs[row_index//stride[0],col_index//stride[1],:,:] =  local_evecs
            
    return local_PCA_evals, local_PCA_evecs

##############################################################
#### Projecting images along vectors and scoring projections######
###################################################


def make_img_proj_arr(vecs, imgs, normalize_imgs=True):
    """
    Make an array of image pojections along a set of vectors
    """
    
    imgs = np.reshape(imgs, newshape=(np.shape(imgs)[0], np.shape(vecs)[0]))
    if normalize_imgs:
        imgs = imgs / np.asarray([LA.norm(imgs[i,:]) for i in xrange(0,np.shape(imgs)[0])])[:,None]
    proj_arr = np.dot(imgs, vecs)
    return proj_arr 
    
def calculate_JS_div_along_PCvecs(vecs, pos_imgs,
                                    neg_imgs, window, stride=(1,1), proj_bins=15):
    """
    return array of shape 
      (ceil((img_rows - window[0] + 1.0)/stride[0])), ceil((img_cols - window[1]  + 1.0)/stride[1]),
      num_vecs)
      
    where returned_arr[i,j,k] is the JS divergence of projections of positive and negtive images 
    restricted to the i, j rectangular window of the image classes onto the kths vector of vecs.
    """
    
    img_rows , img_cols = np.shape(pos_imgs)[-2:]
    assert np.shape(neg_imgs)[-1] == img_cols
    assert np.shape(neg_imgs)[-2] == img_rows

    output_arr_dims =(
                    np.int(np.ceil((img_rows - window[0] + 1.0)/stride[0])), 
                    np.int(np.ceil((img_cols - window[1]  + 1.0)/stride[1]))  
                            ) 

    JS_D_arr=np.zeros(shape =(output_arr_dims[0], output_arr_dims[1], np.shape(vecs)[1]),
                        dtype=np.float)
                    
    for row_index  in xrange(0,img_rows - window[0] + 1, stride[0]):
        for col_index  in xrange(0,img_cols - window[1] + 1, stride[1]):
            local_pos_imgs =pos_imgs[:,row_index:row_index + window[0], col_index:col_index + window[1]]
            pos_imgs_proj = make_img_proj_arr(vecs, local_pos_imgs)
            
            local_neg_imgs = neg_imgs[:,row_index:row_index + window[0], col_index:col_index + window[1]]
            neg_imgs_proj  = make_img_proj_arr(vecs, local_neg_imgs)
            
            bin_edges =np.linspace(-1, 1,num=proj_bins+1)
            pos_imgs_hists = np.hstack([np.histogram(pos_imgs_proj[:,i], bins = bin_edges)[0][:,None]
                                        for i in xrange(0,np.shape(vecs)[1])])
            neg_imgs_hists =np.hstack([np.histogram(neg_imgs_proj[:,i], bins = bin_edges)[0][:,None]
                                        for i in xrange(0,np.shape(vecs)[1])])                         
            
            JS_D_arr[row_index//stride[0],col_index//stride[1],:]=np.asarray(
                    [JS_divergence(pos_imgs_hists[:,i],neg_imgs_hists[:,i]) for i in xrange(0,np.shape(vecs)[1])]
                    )
    return JS_D_arr    
      
def make_PCvec_score_distributions(vecs, pos_imgs, neg_imgs, window, stride=(1,1), proj_bins=15):
    
    img_rows , img_cols = np.shape(pos_imgs)[-2:]
    assert np.shape(neg_imgs)[-1] == img_cols
    assert np.shape(neg_imgs)[-2] == img_rows

    output_arr_dims =(
                    np.int(np.ceil((img_rows - window[0] + 1.0)/stride[0])), 
                    np.int(np.ceil((img_cols - window[1]  + 1.0)/stride[1]))  
                            ) 

    pos_imgs_hists_arr=np.zeros(
                    shape=(output_arr_dims[0], output_arr_dims[1],proj_bins, np.shape(vecs)[-1]),
                    dtype=float)
    neg_imgs_hists_arr =  np.zeros(
                    shape=(output_arr_dims[0], output_arr_dims[1],proj_bins, np.shape(vecs)[-1]),
                    dtype=float)     

    for row_index  in xrange(0,img_rows - window[0] + 1, stride[0]):
        for col_index  in xrange(0,img_cols - window[1] + 1, stride[1]):
            local_pos_imgs =pos_imgs[:,row_index:row_index + window[0], col_index:col_index + window[1]]
            pos_imgs_proj = make_img_proj_arr(vecs, local_pos_imgs)
            local_neg_imgs = neg_imgs[:,row_index:row_index + window[0], col_index:col_index + window[1]]
            neg_imgs_proj  = make_img_proj_arr(vecs, local_neg_imgs)

            bin_edges =np.linspace(-1,1,num=proj_bins +1)
            
            pos_imgs_hists = np.hstack([np.histogram(pos_imgs_proj[:,i], bins = bin_edges)[0][:,None]
                                        for i in xrange(0,np.shape(vecs)[1])])
            pos_imgs_hists_arr[row_index//stride[0],col_index//stride[1],:,:] = pos_imgs_hists
    
            neg_imgs_hists =np.hstack([np.histogram(neg_imgs_proj[:,i], bins = bin_edges)[0][:,None]
                                        for i in xrange(0,np.shape(vecs)[1])])                         
            neg_imgs_hists_arr[row_index//stride[0],col_index//stride[1],:,:] = neg_imgs_hists
      
    return pos_imgs_hists_arr, neg_imgs_hists_arr     
    
    
def JS_divergence(P,Q, pseudo_count=True):
    if pseudo_count:
        pseudo_count=0.01*min(np.mean(P), np.mean(Q))
    else: 
        pseudo_count =0.0
    
    P=P+pseudo_count
    Q = Q+ pseudo_count 
    P = P/np.sum(P)
    Q =Q/np.sum(Q)
    
    M = 0.5*(P+Q)
    JS_D = 0.5*np.dot(P, np.log(np.divide(P,M))) +0.5 * np.dot(Q, np.log(np.divide(Q,M)))
    return JS_D