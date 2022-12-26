# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:03:37 2022

@author: user
"""
#########################################################################################################################
# This code read the "less_than_50" image and connect the gaps in the skeleton
# until we have one connected skeleton. save the image as "connected_skeleton"
#########################################################################################################################
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
from math import hypot
import time

########################parmeters################################################
distance_threshold = 30
#  if needs to get the image from local device
# skel_img = cv2.imread('lessThan30.jpg', 0)
# img = cv2.threshold(skel_img, 127, 255, cv2.THRESH_BINARY)[1]
## creating a dictionary 
mydict ={}#2 edge points we need to conect and the disatance between them
mydict2 ={}#2 edge points we need to conect and the spot's label (bloobs) of each one
mydict3 ={}#all the edges points and thier blob label
mydict4 ={}
coor_and_labels ={}#coordinates and their spot's label 

global first_blob_labels 
first_blob_labels ={}
# For checking the time it takes 
# start_time = time.time()
##################################################################################



def connected_skleton(img,path):
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    plt.imshow(img)
    plt.show()
    num_clustes =len(np.unique(blobs(img)))
    counter = num_clustes
    new_blobs_labels = blobs(img)
    global first_blob_labels 
    first_blob_labels = new_blobs_labels
    white =np.argwhere(img == 255)
    for i in range(white.shape[0]):
         coor_and_labels[white[i][0], white[i][1]] =  new_blobs_labels[white[i][0]][white[i][1]]
    
    new_edges =find_edges(img)
    find_min_dist(new_edges, new_blobs_labels)
    
    #connect two points until counter is less than 2.
    #counter is equal to the connected components in the begining
    #any time we connect between two connceted components we decrease counter by 1
    while counter>2:
            counter = connect_2_points(counter, img)
    
    filename = 'new_try_skeleton2.jpg'
    cv2.imwrite(filename, img)
    plt.imshow(img)
    plt.show()
    print_img(path, img)
    # print("--- %s seconds ---" % (time.time() - start_time))
    return img



                    
  
        
  
    
def clean_skeleton(skel_img, path):
    skel_img = cv2.threshold(skel_img, 127, 255, cv2.THRESH_BINARY)[1]

    new_edges =find_edges(skel_img)

    #run until we left with only 2 edges- head and tail
    while(len(new_edges)>2):   
        new_edges =find_edges(skel_img)

        mydict ={}
        mydict4 ={}
        find_max_dist(new_edges,mydict)
        for key,value in mydict.items():
            first_edge_x = key[0]
            first_edge_y = key[1]
            secondt_edge_x = key[2]
            secondt_edge_y = key[3]
    
    #if the edges are not belongs to the head or tail - paint them in black
        for p in range(new_edges.shape[0]):
            if(not((new_edges[p][0]==first_edge_x and new_edges[p][1]==first_edge_y) or (new_edges[p][0]==secondt_edge_x and new_edges[p][1]==secondt_edge_y))):
                skel_img[new_edges[p][0]][new_edges[p][1]]=0
            # else:
   
        white =np.argwhere(skel_img == 255)
        
        new_edges =find_edges(skel_img)

    # print(path)
    print_img(path, skel_img)
    # filename2 = path+'\new_skeleton.jpg'
    # cv2.imwrite(filename2, skel_img)
    
    return skel_img

### finding connected components
def blobs(img):
    blobs = img
    all_labels = measure.label(blobs)
    blobs_labels = measure.label(blobs, background=0)
    return(blobs_labels)
    

################################################################## 
#finding all the indexes of the white pixels:
# for each white pixel we will find the amount of neighbours so-
# if it has only one neighbour we will consider it to be the edge of a connected component
#or it is an egde point if there is NO two neighbours from the right and left OR two nieghbours from up and buttom  
#if more ->then it is a part of a connected component 
#for each white pixel we wiil check 8 near nieghbours
def find_edges(img):
    white =np.argwhere(img == 255)
    count_neigbours =0
    edge= np.empty((0,2), int)
    for i in range(white.shape[0]):
      up=0
      bottom =0
      right=0
      left=0
      count_neigbours =0 
      
      x =white[i][0]
      y =white[i][1]
      try:
          if(img[x-1][y+1] ==255):
              count_neigbours+=1
              up+=1
              left+=1
      except:
        print("in except")
        continue;
          
      if(img[x][y+1] ==255):
          count_neigbours +=1
          up+=1

      if(img[x+1][y+1] ==255):
          count_neigbours+=1
          up+=1
          right+=1
         
      if(img[x-1][y] ==255):
          count_neigbours+=1
          left+=1  
          
      if(img[x+1][y] ==255):
          count_neigbours+=1
          right+=1
        
      if(img[x-1][y-1] ==255):
          count_neigbours+=1
          bottom+=1
          left+=1 
          
      if(img[x][y-1] ==255):
          count_neigbours+=1
          bottom+=1
       
      if(img[x+1][y-1] ==255):
          count_neigbours+=1
          bottom+=1
          right+=1
          
      if(count_neigbours<=1):
          edge=np.append(edge, np.array([[x,y]]), axis=0)
      if(count_neigbours>1):
#         if the neigbours are not from right and left of the pixel or from up and down -> this is an egde point (with 2 neigbours or more)
          if(not ((right>=1 and left>=1) or (up>=1 and bottom>=1))):
              edge=np.append(edge, np.array([[x,y]]), axis=0)
    return(edge)        
           
  


def Dis(x1,y1,x2,y2):
    dist =((x2-x1)**2+(y2-y1)**2)**.5
    return(dist)

        
def find_min_dist(edge, blobs_labels): 

    min_val =200000

    for p in range(edge.shape[0]):
        x1 = edge[p][0]
        y1 = edge[p][1]

        for j in range(edge.shape[0]):
            if(p!=j):#nor the same point
               
               x2 = edge[j][0]
               y2 = edge[j][1]

               if(blobs_labels[x1][y1]!=blobs_labels[x2][y2]):#the two points are not in the same  spot's label
                   curr =(Dis(x1,y1,x2,y2))
                   mydict4[x1,y1, x2,y2] =  curr


                   mydict[x1,y1, x2,y2] =  curr 
                   mydict2[x1,y1, x2,y2] =  [blobs_labels[x1][y1] , blobs_labels[x2][y2] ]
                   mydict3[x1,y1] =  [blobs_labels[x1][y1]]
        min_val =20000

           
########################

##concect two pionts. in the end of any connection decrase counter by 1 and send it back
def connect_2_points( counter, img):
    new_x= None
    new_y= None
    old_x= None
    old_y= None
    all_pixels = []
    #find the pixels of the min distance
    min_points = min(mydict, key=mydict.get)
    #find the distance between the points
    minDist = mydict[min_points[0],min_points[1],min_points[2],min_points[3]]
    if(mydict3[min_points[0],min_points[1]]==mydict3[min_points[2],min_points[3]]): #if the 2 edges are in the same blob - remove them from mydict and return
        try:
            del mydict[min_points[0],min_points[1],min_points[2],min_points[3]]
        except:

             print("problem to remove1") 
        try:
            del mydict[min_points[2],min_points[3],min_points[0],min_points[1]]
        except:
            print("problem to remove2") 
        return (counter)
    #connect the components by adding pixel everytime until the distance between the components is less than 2**0.5
    while(minDist>2**0.5):
        if new_x != None: #since the second time
            
            x1 =new_x
            y1 =new_y
            x2 =old_x
            y2 =old_y
        else: #for the first time
           
            x1 =min_points[0]
            y1 =min_points[1]
            x2 =min_points[2]
            y2 =min_points[3]

        #if the distance is longer than the threshold - it means that we dont need to connect any more. 
        #we will loop all over the white pixels that left and remove all of them except the pixels of the real skeleton
        #whice is the largest conncected component
        if(mydict[min_points]>distance_threshold):
            new_blobs_labels = blobs(img)
            all_white_pixels =np.argwhere(img == 255)
            for i in range(all_white_pixels.shape[0]):
                coor_and_labels[all_white_pixels[i][0], all_white_pixels[i][1]] =  new_blobs_labels[all_white_pixels[i][0]][all_white_pixels[i][1]]
            largest_component =0
            #find the largest connected component (must be the desired skeleton)
            for i in range (len(np.unique(first_blob_labels))):
                label_1 = [k for k, v in coor_and_labels.items() if v == i]
                if(len(label_1)>largest_component):
                    largest_component = len(label_1)
                    largest_blob = i
                    all_pixels = label_1

            filename = 'new_try_before.jpg'
            cv2.imwrite(filename, img)
            #remove all the white pixels if they are not part of the largest connected component
            for i in all_white_pixels:
                if(not any((i == x).all() for x in all_pixels)):
                    img[i[0]][i[1]]=0#make it black
            #assign counter to 1 which means the end of the main while loop 
            counter=1
            return (counter)
        #connect connected componnents
        delthaX =abs(x1-x2)
        delthaY =abs(y1-y2)

        if(delthaX<delthaY):
            if(delthaX == 0):
                new_x = x1
                old_x = x2
            else:
                if(min(y1 , y2)==y1):
                    if(x1>x2):
                        new_x =round(x1 - (delthaX/delthaY))
                        old_x = x2
                    else:
                        new_x =round(x1 + (delthaX/delthaY))
                        old_x = x2
                else: # min(y1 , y2)==y2
                    if(x1<x2):
                        new_x =round(x2 - (delthaX/delthaY))
                        old_x = x1
                    else:
                        new_x =round(x2 + (delthaX/delthaY))
                        old_x = x1

            new_y =round(min(y1 , y2) + 1)
            old_y = round(max(y1 , y2))


            while img[new_x][new_y] ==255:

                if(img[new_x][new_y] ==255):
                    new_y+=1

                
    # deltaY<deltaX
        else:
            if(delthaY == 0):
                new_y = y1
                old_y = y2
            else:
                if(min(x1 , x2)==x1):
                    if(y1>y2):
                        new_y =round(y1 - (delthaY/delthaX))
                        old_y = y2
                    else:
                        new_y =round(y1 + (delthaY/delthaX))
                        old_y = y2
                else: # min(x1 , x2)==x2
                    if(y1<y2):
                        new_y =round(y2 - (delthaY/delthaX))
                        old_y = y1
                    else:
                        new_y =round(y2 + (delthaY/delthaX))
                        old_y = y1


            new_x = round(min(x1 , x2) + 1)
            old_x = round(max(x1 , x2))
            while img[new_x][new_y] ==255:
                if(img[new_x][new_y] ==255):
                    new_x+=1 
        #coloring the new point    
        img[new_x][new_y] =255

        minDist = Dis(new_x,new_y,old_x,old_y)

#     search all the edges belongs to the same blob and insert a new blob label
    value = {i for i in mydict3 if mydict3[i]==mydict3[min_points[0],min_points[1]]}

    for i in value:
        mydict3[i[0],i[1]] = mydict3[min_points[2],min_points[3]]
    #decrease counter by 1 because we connected two parts
    counter= counter-1
    
 
    try:
        
        del mydict[min_points[0],min_points[1],min_points[2],min_points[3]]
    except:

         print("problem") 
    try:
        del mydict[min_points[2],min_points[3],min_points[0],min_points[1]]
    except:

         print("problem")  
    return (counter)

# find the max distance between pixels and put it in mydict
def find_max_dist(edge,mydict): 

    max_val =0

    for p in range(edge.shape[0]):
        x1 = edge[p][0]
        y1 = edge[p][1]
        for j in range(edge.shape[0]):
            if(p!=j):#not the same point
               x2 = edge[j][0]
               y2 = edge[j][1]
              
               curr =(Dis(x1,y1,x2,y2))
               # mydict4[x1,y1, x2,y2] =  curr
               if(curr> max_val):
                    max_val =curr
                    new_x2 =x2
                    new_y2 =y2
                    mydict.clear()
                    mydict[x1,y1, new_x2,new_y2] =  max_val
    





def print_img(path, skel_img):
    filename2 = path+'\new_skeleton.jpg'
    cv2.imwrite(filename2, skel_img)
    print("in the function of  print clean skeleton")
