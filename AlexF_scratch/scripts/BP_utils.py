### copied from https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/data.py
### by Marko Jocic


from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import  linalg as LA
import networkx as nx
from scipy import linalg as spLA

data_path = 'raw/'

image_rows = 420  # how did marko get the image dims ?
image_cols = 580

### These functions shouldn be run from directory with the following subdirs
## raw/train  with unzipped training data 
## raw/test   with unzipped test data

def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) / 2

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')
    

    
def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train
    
    

def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

#if __name__ == '__main__':
#    create_train_data()
#    create_test_data()

### code copied from
# https://www.kaggle.com/chefele/ultrasound-nerve-segmentation/plot-images-overlaid-with-mask
# by christopher Hefele

def image_with_mask(img, mask):
    # returns a copy of the image with edges of the mask added in red
    img_color = grays_to_RGB(img)
    mask_edges = cv2.Canny(mask, 100, 200) > 0  
    img_color[mask_edges, 0] = 255  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0
    return img_color

def mask_not_blank(mask):
    return sum(mask.flatten()) > 0

def grays_to_RGB(img):
    # turn 2D grayscale image into grayscale RGB
    return np.dstack((img, img, img)) 

def plot_image(img, title=None,):
    plt.figure(figsize=(15,20))
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()
    
## Alex's functions ######################


def plot_mask_with_decision_bndry(ax, img_arr, mask_arr, decision_bndry=None,
                                    linewidth=2):
    ax.imshow(image_with_mask(img_arr,mask_arr))
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    
    if decision_bndry is not None:
        xvals =decision_bndry[0]
        yvals=decision_bndry[1]
        ax.plot(xvals,yvals, linewidth=linewidth)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
            
    #return ax
        
def plot_example(image_arr, mask_arr, offset=-0.5,figsize=(14.0,14.0) ):
    
    min_x=offset
    max_x =np.shape(image_arr)[1] + offset
    min_y = offset
    max_y = np.shape(image_arr)[0] + offset
    
    #create dummy decision bndry data
    x_center=np.random.uniform(min_x + (max_x-min_x)/4.0, 
                                    min_x + 3*(max_x-min_x)/4.0 )
    y_center=np.random.uniform(min_y + (max_y-min_y)/4.0,
                                    min_y + 3*(max_y-min_y)/4.0 )
    radius=np.random.uniform(0.,(max_x-min_x)/4.0)
    

    decision_bndry = (radius*np.cos(np.linspace(0,2*np.pi,100))+ x_center,
                        radius*np.sin(np.linspace(0,2*np.pi,100))+ y_center)
    
    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot('111')
    ax = plot_mask_with_decision_bndry(ax,image_arr, mask_arr, decision_bndry)
    return fig
    

def mask_from_circle_params(center, radius, img_shape=(580,420)): 
    """
    returns a 2d numpy boolean array that is true whenever a pixel is within the 
    circle specified by center= (x_center ,y_center) and radius and False otherwise.
    
    Inputs
    center
    radius
    image_shape=(number of pixels along x (horizontal) axis, number of pixels along y (vertical) axis)
    """
# http://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array   
    y_vals, x_vals = np.ogrid[0:img_shape[1],0:img_shape[0]]
    mask=(x_vals-center[0])**2 + (y_vals-center[1])**2 <= radius**2
    return mask
    
    
        
def dice_coef(y_true, y_pred):
    
    smooth=1 ## used by J marko to enforce condietion dice_coeff -> 1 for no BP nerve
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    
#### Preprocessing code        
                
# PCA code 

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

def make_img_proj_arr(vecs, imgs, normalize_imgs=True):
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
   
### shift and images
    
    

## misc functions 
def resize_images(imgs,img_rows, img_cols, interp_method='INTER_AREA'):
    interp_dict ={'INTER_AREA':cv2.INTER_AREA,
                'INTER_CUBIC': cv2.INTER_CUBIC,
                'INTER_LINEAR':cv2.INTER_LINEAR}

    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), 
            interpolation=interp_dict[interp_method])
    return imgs_p

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