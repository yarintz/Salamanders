# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:59:41 2022

@author: user
"""


########################################################################################################################################
# this code reads the original image(from apper), make morphology (save image as "After_morphology") 
#and makes the first skeleton and save it as "After_skeletonize"
# Import the necessary libraries
##################################################################################################################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

####################################### parameters ####################################
# kernel_par = 15 #size of kernel 
######################################################################################

def create_skeleton (img ,ker,path):

    ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        # closing holes
    kernel = np.ones((ker,ker),np.uint8)
    
    img  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    after_morphology  = np.copy(img)
    filename = path+'\After_morphology.jpg'
    print("filename is: "+ filename)
    cv2.imwrite(filename, after_morphology)
    
    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    
    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    
    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img)==0:
       
            break
    
    # save the image under the name "savedImage"
    filename = path+'\After_skeletonize.jpg'
    cv2.imwrite(filename, skel)
    print("finished morph and sel")
    return skel, after_morphology

# body_img = cv2.imread(r'C:\Users\user\OneDrive - University of Haifa\Desktop\Project\new_salmanders\AH181967\AH181967_background_predictions.ome.jpg',0)
# p = r'C:\Users\user\OneDrive - University of Haifa\Desktop\Project\new_salmanders\AH181967'

# skeleton, morphology = create_skeleton(body_img, 15,p)