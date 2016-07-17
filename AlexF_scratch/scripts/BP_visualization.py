from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import  linalg as LA
import networkx as nx
from scipy import linalg as spLA



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