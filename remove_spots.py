# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 19:51:57 2022

@author: yarin
"""
import numpy as np
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.ensemble import RandomForestClassifier
import cv2
from queue import PriorityQueue
from scipy import ndimage
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
from collections import Counter
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from scipy import ndimage


def remove_spots(skeleton,only_spots, path):
    edt_par = 350
  #  only_spots = cv2.imread('AH171823_2_Yellow1_predictions.ome.jpg', 0)
    #skeleton = cv2.imread('new_skeleton.jpg', 0)
    
    skeleton = 255-skeleton #changing image colors 
    edt, inds = ndimage.distance_transform_edt(skeleton, return_indices=True)
    
    only_spots_copy = np.copy(only_spots)
    white_spots =np.argwhere(only_spots_copy == 255)
    for x in range(only_spots_copy.shape[0]):
        for y in range(only_spots_copy.shape[1]):
            if(edt[x,y]>edt_par):
                only_spots_copy[x,y] = 0
    filename5 = path+'\only_spots_new.jpg'
    cv2.imwrite(filename5, only_spots_copy)
    return only_spots_copy
# skeleton = cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH181965\AH181965_1\new_skeleton.jpg',0)
# spots =cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH181965\AH181965_1\only_spots_new.jpg',0)
# print(skeleton.shape[0])
# print(spots.shape[0])
# remove_spots(skeleton, spots, r'C:\Users\yarin\Documents\salamanders\AH181965\AH181965_1' )