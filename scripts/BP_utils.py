### copied from https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/data.py
### by Marko Jocic


from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

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

def plot_image(img, title=None):
    plt.figure(figsize=(15,20))
    plt.title(title)
    plt.imshow(img)
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
        
        
        
        