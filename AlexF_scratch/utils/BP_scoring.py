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



def dice_coef(y_true, y_pred):
    
    smooth=1 ## used by J marko to enforce condietion dice_coeff -> 1 for no BP nerve
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    
  