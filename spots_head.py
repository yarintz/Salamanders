# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 12:02:53 2022

@author: user
"""
from new_skeleton import create_skeleton
from remove_legs_new import remove_legs_from_skeleton
import connectcomponents as connect
from find_head import head_tail
import graph_svd_new as svd
import cv2
#import spots 
import numpy as np
import pickle
import salamander_spots_class as ssc
import numpy as np
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
import cv2
from skimage import measure
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
# import spots 
import numpy as np





# skeleton =sal1.final_skel
# spots =sal1.spots_img
# spots = cv2.threshold(spots, 127, 255, cv2.THRESH_BINARY)[1]

# def create_blobs(img):
#     blobs = img
#     all_labels = measure.label(blobs)
#     blobs_labels = measure.label(blobs, background=0)
#     return(blobs_labels)

# blobs =create_blobs(spots)

    
# filename = 'TRY_SKELETON.jpg'
# cv2.imwrite(filename, skeleton) 

# filename = 'TRY_Spots.jpg'
# cv2.imwrite(filename, spots) 

import cv2
import numpy as np
import math
from skimage import measure
import matplotlib.pyplot as plt
from queue import PriorityQueue

def find_edges(skeleton_white, skeleton):
    count_neigbours =0
    edge= np.empty((0,2), int)
    list_edges = []
    for i in range(skeleton_white.shape[0]):

        up=0
        bottom =0
        right=0
        left=0
        count_neigbours =0 

        x =skeleton_white[i][0]
        y =skeleton_white[i][1]
        if(skeleton[x-1][y+1] ==255):
          count_neigbours+=1
          up+=1
          left+=1

        if(skeleton[x][y+1] ==255):
          count_neigbours +=1
          up+=1
        
        if(skeleton[x+1][y+1] ==255):
          count_neigbours+=1
          up+=1
          right+=1

        if(skeleton[x-1][y] ==255):
          count_neigbours+=1
          left+=1  

        if(skeleton[x+1][y] ==255):
          count_neigbours+=1
          right+=1

        if(skeleton[x-1][y-1] ==255):
          count_neigbours+=1
          bottom+=1
          left+=1 

        if(skeleton[x][y-1] ==255):
          count_neigbours+=1
          bottom+=1

        if(skeleton[x+1][y-1] ==255):
          count_neigbours+=1
          bottom+=1
          right+=1

        if(count_neigbours<=1):
          edge=np.append(edge, np.array([[x,y]]), axis=0)
        if(count_neigbours>1):
        #         if the neigbours are not from right and left of the pixel or from up and down -> this is an egde point (with 2 neigbours or more)
          if(not ((right>=1 and left>=1) or (up>=1 and bottom>=1))):
              edge=np.append(edge, np.array([[x,y]]), axis=0)
              list_edges.append(edge) 
    return(edge)
# skeleton = cv2.imread('connected_skeleton.jpg', 0)
# skeleton = cv2.threshold(skeleton, 127, 255, cv2.THRESH_BINARY)[1]
# spots_image = cv2.imread('AH181967_Yellow1_predictions.ome.jpg', 0)
# spots_image = cv2.threshold(spots_image, 127, 255, cv2.THRESH_BINARY)[1]
#get all the white pixels of the skeleton
def return_head_tail(skeleton, spots):
    q = PriorityQueue()
    q.queue.clear()
    skeleton = cv2.threshold(skeleton, 127, 255, cv2.THRESH_BINARY)[1]
    skeleton_white =np.argwhere(skeleton == 255)
    spots = cv2.threshold(spots, 127, 255, cv2.THRESH_BINARY)[1]
    only_spots = np.argwhere(spots == 255)
    coor_and_labels ={}#coordinates and their spot's label 
    
    blobs = spots
    all_labels = measure.label(blobs)
    blobs_labels = measure.label(blobs, background=0)
    edges = find_edges(skeleton_white, skeleton)
    # fill the dictionary:
    # key - all the coordinates of the white pixels
    # value - the blob label of the coordinates
    for i in range(only_spots.shape[0]):
         coor_and_labels[only_spots[i][0], only_spots[i][1]] =  blobs_labels[only_spots[i][0]][only_spots[i][1]]
    
    for i in range (len(np.unique(blobs_labels))):
        label_1 = [k for k, v in coor_and_labels.items() if v == i]
        #print(str(i)+" "+str(len(label_1)))
    #     the queue holds the num of white pixels by increasing order and the number of the blob label
        q.put((len(label_1)*-1, i))
        
        
        
    
    max_items=[]
    for i in range(len(np.unique(blobs_labels))-1):
        next_item = q.get()[1]
        label_2 = [k for k, v in coor_and_labels.items() if v == next_item]
        #print(label_2[0])
        max_items.append(label_2[0])
    # print("max_items")
    # print(max_items)
    count1= 0
    count2=0
    i=0;
    while (count1<2 and count2<2):
        i=i+1
        if(i == len(max_items)):
            print("end of max item")
            break;
        spot = max_items[i]
        # print("spot")
        # print(spot)
        #calculate the distance between the biggest_spot to the edges of the skeleton - the closer is the head and the other is the tail
        distance_1 = math.dist([spot[0],spot[1]], [edges[0][0],edges[0][1]])
        distance_2 = math.dist([spot[0],spot[1]], [edges[1][0],edges[1][1]])
    
        if(distance_1 <200 or distance_2<200):
            if distance_1<distance_2:
                count1= count1+1
            else:
                count2= count2+1
    
    
    if count1>count2:
        head = edges[0]
        tail = edges[1]
    else:
        head = edges[1]
        tail = edges[0]
    print("-----------")
    print(head)
    print(tail)
    return head, tail