# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:38:35 2022

@author: user
"""

##################################################################################################################
#          remove the legs from the "After_skeletonize" image with data transfrom on the "after_morphology" image.
#          save a new image as "lessThan50"
##################################################################################################################
import cv2
from scipy import ndimage
from matplotlib import pyplot as plt
import numpy as np


def remove_legs_from_skeleton(img_morphology,img_skel,edt_par,path):
    #image tranfron -  edt all distances from nearst black pixels
    # inds - the nearst black point (x,y values) from each point 
    edt, inds = ndimage.distance_transform_edt(img_morphology, return_indices=True)
    #for each pixel in the skleton, if it's edt value(in the after morphology image) is less than the threshold->we will change it to black
    #original skeleton we will use for the rest of the process
    img_original_skel = np.copy(img_skel)
    for x in range(img_original_skel.shape[0]):
        for y in range(img_original_skel.shape[1]):
            if(edt[x,y]<edt_par):
                img_original_skel[x,y] = 0     
    # plt.imshow(img_original_skel)
    filename = path+'\lessThan'+str(edt_par)+'.jpg'
    cv2.imwrite(filename, img_original_skel)
    return img_original_skel


