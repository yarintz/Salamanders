# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:28:12 2022

@author: user
"""
import cv2
import numpy as np
import math
from skimage import measure
from matplotlib import pyplot as plt
# get a long skeleton image and a shorter one
#long_skeleton = cv2.imread(r'C:\Users\user\OneDrive - University of Haifa\Desktop\Project\ahsalmander\connected_skeleton30.jpg', 0)
#short_skeleton = cv2.imread(r'C:\Users\user\OneDrive - University of Haifa\Desktop\Project\7_salmanders_AH171801\AH171801_1\New_version\connected_skeleton_new_12_less_25_goodluck.jpg', 0)

def head_tail(long_skeleton,short_skeleton ):
    long_skeleton = cv2.threshold(long_skeleton, 127, 255, cv2.THRESH_BINARY)[1]
    short_skeleton = cv2.threshold(short_skeleton, 127, 255, cv2.THRESH_BINARY)[1]
    #get all the white pixels
    long_skeleton_white =np.argwhere(long_skeleton == 255)
    short_skeleton_white =np.argwhere(short_skeleton == 255)
    
    coor_and_labels ={}#coordinates and their spot's label 
    plt.imshow(long_skeleton)
    plt.show()
    plt.imshow(short_skeleton)
    plt.show()
    # make XOR between the two images
    # new_img = np.bitwise_xor(long_skeleton, short_skeleton)
    
    new_img = long_skeleton ^ short_skeleton
    new_img = cv2.threshold(new_img, 127, 255, cv2.THRESH_BINARY)[1]
    
    filename = r'C:\Users\yarin\Documents\salamanders\AH171823\AH171823_5\head_tail.jpg'
    cv2.imwrite(filename, new_img)
    
    #get the white pixels of the new image
    new_skeleton_white =np.argwhere(new_img == 255)
    
    # find the connected componnents - there must be 2:
    # the tail - the long one
    # the head - the short one
    blobs = new_img
    all_labels = measure.label(blobs)
    blobs_labels = measure.label(blobs, background=0)
    
    # make sure there are only 2 clusters(not including the background)
    num_clusters =len(np.unique(blobs_labels))
    print("num of blobs ", num_clusters)
    
    # fill the dictionary:
    # key - all the coordinates of the white pixels
    # value - the blob label of the coordinates
    for i in range(new_skeleton_white.shape[0]):
          coor_and_labels[new_skeleton_white[i][0], new_skeleton_white[i][1]] =  blobs_labels[new_skeleton_white[i][0]][new_skeleton_white[i][1]]
    
    #assign the coordinates to new values by their blob label 
    label_1 = [k for k, v in coor_and_labels.items() if v == 1]
    print(len(label_1))
    label_2 = [k for k, v in coor_and_labels.items() if v == len(np.unique(blobs_labels))-1] 
    #check which label is smaller - the smaller is the head
    #assign to temp_head the value of a point belongs to the shorter label
    if(len(label_1)<len(label_2)): 
        temp_head = label_1[0] 
    else:
        temp_head = label_2[0]
    
    #calculate the distance between the temp_head to the first and last points of the skeleton of the original image
    distance_1 = math.dist([temp_head[0],temp_head[1]], [short_skeleton_white[0][0],short_skeleton_white[0][1]])
    print(distance_1)
    distance_2 = math.dist([temp_head[0],temp_head[1]], [short_skeleton_white[len(short_skeleton_white)-1][0],short_skeleton_white[len(short_skeleton_white)-1][1]])
    print(distance_2)
    
    #determine the head to be the edge point which is closer to the temp_head
    if distance_1<distance_2:
        head = short_skeleton_white[0]
        tail = short_skeleton_white[len(short_skeleton_white)-1]
    else:
        head = short_skeleton_white[len(short_skeleton_white)-1]
        tail = short_skeleton_white[0]
    print("head----")
    print(head)
    print("tail----")
    print(tail)
    # def head_tail():x
    return head,tail
    # h1,t1 =head_tail()
    
# h1,t1 =head_tail(long_skeleton,short_skeleton)