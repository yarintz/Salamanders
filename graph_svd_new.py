# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 13:15:13 2022

@author: user
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:44:57 2022

@author: user
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
# from find_head import head_tail
import pickle 

# sala = pickle.load( open("AH181965_1.p", "rb" ) )
# salb = sala.copy()
# img =salb.final_skel
# spots_img =salb.spots_img
# morph_img = salb.morphology
# head = salb.head
# tail = salb.tail
# path = salb.path
# spots_img = cv2.imread(r'C:\Users\user\OneDrive - University of Haifa\Desktop\Project\7_salmanders_AH171801\AH171801_1\New_version\connected_skeleton_new_12_less_25_goodluck.jpg', 0)

# long_skeleton = cv2.imread(r'C:\Users\user\OneDrive - University of Haifa\Desktop\Project\ahsalmander\connected_skeleton30.jpg', 0)
# short_skeleton = cv2.imread(r'C:\Users\user\OneDrive - University of Haifa\Desktop\Project\7_salmanders_AH171801\AH171801_1\New_version\connected_skeleton_new_12_less_25_goodluck.jpg', 0)


def run_svd(img, spots_img, morph_img, kernel_par,head, tail,path):
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    img_spots = cv2.threshold(spots_img, 127, 255, cv2.THRESH_BINARY)[1]
    morph1 =cv2.threshold(morph_img, 127, 255, cv2.THRESH_BINARY)[1]
    morph =np.argwhere(morph1 == 255)

    kernel_par = kernel_par  #size of kernel 
    head =head
    tail =tail
    # connected_skeleton = cv2.imread(r'C:\Users\user\OneDrive - University of Haifa\Desktop\Project\ahsalmander\check.jpg', 0)
    # img_spots = cv2.imread(r'C:\Users\user\OneDrive - University of Haifa\Desktop\Project\7_salmanders_AH171801\AH171801_1\New_version\AH171801_Yellow1_predictions.ome.jpg', 0)
    
    ######################paraneters############################################################################################################
    # kernel_par = 5 #size of kernel 
    ###################################################################################################################
    # #print
    # f = open(r'C:\Users\user\OneDrive - University of Haifa\Desktop\Project\ahsalmander\salamander_info.txt', 'a')
    # f.write('kernel_par ')
    # f.write(str(kernel_par))
    # f.write('\n')
    # f.close()
    ########################################################################################################################################################################################################################################
    ##loading image
    # connected_skeleton = cv2.imread(r'C:\Users\user\OneDrive - University of Haifa\Desktop\Project\ahsalmander\check.jpg', 0)
    # img = cv2.threshold(connected_skeleton, 127, 255, cv2.THRESH_BINARY)[1]
    
    #spots image
    # img_spots = cv2.imread(r'C:\Users\user\OneDrive - University of Haifa\Desktop\Project\7_salmanders_AH171801\AH171801_1\New_version\AH171801_Yellow1_predictions.ome.jpg', 0)
    # img_spots = cv2.threshold(img_spots, 127, 255, cv2.THRESH_BINARY)[1]
    ####################################################################################################################################################################################################################################3
    #numbering all the spots 
    ### finding connected components
    def blobs(img):
        blobs = img
        all_labels = measure.label(blobs)
        blobs_labels = measure.label(blobs, background=0)
        return(blobs_labels)
    spots =blobs(img_spots)
    ####################################################################
    # spots_group ={}
    # spots_new_img ={}
    # help_spots ={}
    # for i in range(spots.shape[0]):
    #     for j in range(spots.shape[0]):
    #         if(spots[i][j]!=0):
    #             spots_group[i,j] =spots[i][j]
    ###############################################################
    # setting new array of only the white pixels from the spots image
    only_spots = np.empty((0,2), int)
    for x in range(img_spots.shape[0]):
        for y in range(img_spots.shape[1]):
            if(img_spots[x,y]!=0):
                #only_spots - x,y values array of all the spots
                only_spots = np.append(only_spots, np.array([[x,y]]), axis=0)
    spots_group ={}
    spots_new_img ={}
    help_spots ={}
    pixel_numOfNode={} #key = a pixel of the skeleton, value = the number of node from the graph
    for i in range(only_spots.shape[0]):
        #            Key: x,y of the spot                        Value - spot's group (from the blobs)
          spots_group[only_spots[i][0],only_spots[i][1]] =spots[only_spots[i][0]][only_spots[i][1]] 
    
    #################################################################################################################################################################################
    only_white = np.empty((0,2), int)# connected_skeleton -white pixels
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if(img[x,y]!=0):
                only_white = np.append(only_white, np.array([[x,y]]), axis=0)
    
    white =np.copy(only_white)  
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #   SVD
    mean=np.mean(only_white, axis=0)
    only_white[:,0] =only_white[:,0] -mean[0]
    only_white[:,1] =only_white[:,1] -mean[1]
    U, sigma, V = np.linalg.svd(only_white , full_matrices=False)
    

    # head_tail =tail-head
    # print("head_tail ", head_tail)
    # #v1* head_tail
    # sign_v1 =np.dot(head_tail,V[0,:])
    # print("sign_v1", sign_v1)

    # if(sign_v1<0):
    #     V[0,:]= V[0,:]*-1
    # sign_v2 = np.cross(V[0,:],V[1,:])
    # print("sign_v2", sign_v2)
    # if(sign_v2<0):
    #     V[1,:]= V[1,:]*-1
 
    head_tail =tail-head

    #v1* head_tail
    sign_v1 =np.dot(head_tail,V[0,:])
    print("sign_v1", sign_v1)
    print(V[0])
    if(sign_v1<0):
        V[0,:]= V[0,:]*-1
    print("V[0]")    
    print(V[0])

    sign_v2 = np.cross(V[0,:],V[1,:])
    print("sign_v2", sign_v2)
    if(sign_v2<0):
        V[1,:]= V[1,:]*-1
    ################################################################################################################
    #crating a graph

    dist_in_pixels ={}
    
    for i in range(white.shape[0]):
        #KEY x,y                          VALUE pixel's number
        dist_in_pixels[white[i][0], white[i][1]] = i
    
    def Dis(x1,y1,x2,y2):
        dist =((x2-x1)**2+(y2-y1)**2)**.5
        return(dist)
    
    # #finding the size of the edges
    # pixel_graph =[]
    # for i in range(white.shape[0]):
    #         for j in range(white.shape[0]):
    #             if(i<j):
    #                 dist =Dis(white[i][0], white[i][1], white[j][0], white[j][1])
    #                 if(dist<=2**.5):
    #                     pixel_graph.append([i,j,dist]) 
    ################################
    
    #starting with the head 
    pixel_graph =[]

        ##finding all pixels nieghbuors 
    for i in range(white.shape[0]):
       for j in range(white.shape[0]):
             if(i==white.shape[0]-1):
                 print("last time ",white[i][0], white[i][1])
                 pixel_numOfNode[(white[i][0], white[i][1])]=i
                 break;
             if(i!=j):
                 dist =Dis(white[i][0], white[i][1], white[j][0], white[j][1])
                 if(dist<=2**.5):
                     #pixels number, pixels number, distance between them
                      pixel_graph.append([i,j,dist]) 
                      pixel_numOfNode[(white[i][0], white[i][1])]=i #the pixel and the number of the node
                     
    
    class Graph:
        def __init__(self, num_of_vertices):
            self.v = num_of_vertices
            self.edges = [[-1 for i in range(num_of_vertices)] for j in range(num_of_vertices)]
            self.visited = []
    
        def add_edge(self, u, v, weight):
            self.edges[u][v] = weight
            self.edges[v][u] = weight
        
        
        def dijkstra(self, start_vertex):
                D = {v:float('inf') for v in range(self.v)}
                D[start_vertex] = 0
            
                pq = PriorityQueue()
                pq.put((0, start_vertex))
            
                while not pq.empty():
                    (dist, current_vertex) = pq.get()
                    self.visited.append(current_vertex)
            
                    for neighbor in range(self.v):
                        if self.edges[current_vertex][neighbor] != -1:
                            distance = self.edges[current_vertex][neighbor]
                            if neighbor not in self.visited:
                                old_cost = D[neighbor]
                                new_cost = D[current_vertex] + distance
                                if new_cost < old_cost:
                                    pq.put((new_cost, neighbor))
                                    D[neighbor] = new_cost
                return D
    
    ##create graph from all the white pixels
    g = Graph(white.shape[0])
    for i in range(len(pixel_graph)):
        g.add_edge(pixel_graph[i][0], pixel_graph[i][1], pixel_graph[i][2])
        
    #Dijkstra ->need to start from the salamnder's head 
    D = g.dijkstra(pixel_numOfNode[(head[0],head[1])])

    pixels_dist_from_head = {}
    pixels_dist_from_head_copy ={}
    for i in range(white.shape[0]):
        # print(white[i])
        x =white[i][0]
        y=white[i][1]
        try:
            node = pixel_numOfNode[x,y]
            pixels_dist_from_head[x,y] =D[node]
            #            pixel- x,y         distance from the head
            pixels_dist_from_head_copy[x,y] =D[node]*.3
        except:
            continue;

    
    ########################################################################################################
    #                                          Distance transfrom
    ###########################################################################################################
    #crating new image of striaghtened salamander
    img_spots
    inds_valuess ={}
    inds_num ={}
    # setting new array of only the white pixels from the spots image
    only_spots = np.empty((0,2), int)
    for x in range(img_spots.shape[0]):
        for y in range(img_spots.shape[1]):
            if(img_spots[x,y]!=0):
                only_spots = np.append(only_spots, np.array([[x,y]]), axis=0)
    ##runnig distance transform on the  connected skeleton 
    white_skel = 255-img
    edt, inds = ndimage.distance_transform_edt(white_skel, return_indices=True)
    #striaghtend salamander
    new_img = np.empty((0,2), int)
    # inds [X or Y values][row][column]
    spots_edt ={}
    #removing the spots
    for i in range(only_spots.shape[0]):
        new_inds =[inds[0][only_spots[i][0]][only_spots[i][1]] ,inds[1][only_spots[i][0]][only_spots[i][1]]]
        # KEY -spot x,y                          VALUE- nearest point from the skeleton
        inds_valuess[only_spots[i][0], only_spots[i][1]] =new_inds
    ############################################################################################
    #                      FINDING THE TALI'S SPOTS  
    ###########################################################################################  
    # list of all the X,Y of inds in the spots    
    inds_list = list(inds_valuess.values())
    
    max_frequency =0
    point_of_inds =[]
    x_inds =0
    y_inds =0
    #finding the most frequent x,y values of the skeleton from the inds list 
    #it's the value of all the tail's spots
    for i in range(len(inds_list)):
        if(max_frequency < inds_list.count(inds_list[i])):
            max_frequency = inds_list.count(inds_list[i])
            x_inds=inds_list[i][0]
            y_inds =inds_list[i][1]
            
            
    tail_spots =[]
    tail_spots = [k for k, v in inds_valuess.items() if v[0] == x_inds and v[1] ==y_inds]
    
    img_spots_no_tail =np.copy(img_spots)
    #removing the tail's spots
    for i in range (len(tail_spots)):
        img_spots_no_tail[tail_spots[i][0]][tail_spots[i][1]]  = 0
    
    filename_img_spots_no_tail ='img_spots_no_tail.jpg'
    cv2.imwrite(filename_img_spots_no_tail, img_spots_no_tail)    
    #######################MORPH#############################################################################################################################
   ######################MORPH#############################################################################################################################
    print("start morph")
    inds_valuess_morph ={}
    for i in range(morph.shape[0]):
        # new_inds =[inds[0][morph[i][0]][morph[i][1]] ,inds[1][morph[i][0]][morph[i][1]]]
        # KEY -MORPH x,y                          VALUE- nearest point from the skeleton
        inds_valuess_morph[morph[i][0], morph[i][1]] = [inds[0][morph[i][0]][morph[i][1]] ,inds[1][morph[i][0]][morph[i][1]]] 
    #
    morph_edt = {}
    help_morph ={}
    new_img_morph = np.empty((0,2), int)
    morph_old_new ={}
    
    for i in range(morph.shape[0]):
        new_edt_morph = edt[morph[i][0],morph[i][1]]
    #                  X value                                     Y value
        new_inds_morph =[inds[0][morph[i][0]][morph[i][1]] ,inds[1][morph[i][0]][morph[i][1]]]
        
    
        pixel_morph = [morph[i][0],morph[i][1]]
        
        sign_morph = np.dot((np.array(pixel_morph) -np.array(new_inds_morph)),V[1,:])
        
        if(sign_morph<0):
            new_edt_morph*=-.3
        else:
            new_edt_morph*=.3
            
        morph_edt[morph[i][0],morph[i][1]] = new_edt_morph #edt value
        
        #the point on the skeleton: 
        x_morph = inds[0][morph[i][0]][morph[i][1]]
        y_morph= inds[1][morph[i][0]][morph[i][1]]
        
        #*KEY: distance from the head, new edt(distance from skeleton)  *VALUE: spot's group
        # help_spots[pixels_dist_from_head_copy[x,y],new_edt] = spots_group[white_spots_no_tail[i][0],white_spots_no_tail[i][1]]
      
        try:
            #                                      # X-distance from head        Y-distance from skeleton
            new_img_morph = np.append(new_img_morph, np.array([[ pixels_dist_from_head_copy[x_morph,y_morph],new_edt_morph]]), axis=0)  
        except:
            continue;
        morph_old_new[morph[i][0],morph[i][1]] = [pixels_dist_from_head_copy[x_morph,y_morph],new_edt_morph]
        
        # morph_old_new[morph[i][0],morph[i][1]] = [pixels_dist_from_head_copy[x_morph,y_morph],new_edt_morph]
    
    
    
       
    #########################################################################################################################
    white_spots_no_tail =np.argwhere(img_spots_no_tail == 255)
    
    spots_old_new ={}
    for i in range(white_spots_no_tail.shape[0]):
        new_edt = edt[white_spots_no_tail[i][0],white_spots_no_tail[i][1]]
    #                  X value                                     Y value
        new_inds =[inds[0][white_spots_no_tail[i][0]][white_spots_no_tail[i][1]] ,inds[1][white_spots_no_tail[i][0]][white_spots_no_tail[i][1]]]
        
    
        pixel = [white_spots_no_tail[i][0],white_spots_no_tail[i][1]]
        
        sign = np.dot((np.array(pixel) -np.array(new_inds)),V[1,:])
        if(sign<0):
            new_edt*=-.3
        else:
            new_edt*=.3
            
        spots_edt[white_spots_no_tail[i][0],white_spots_no_tail[i][1]] = new_edt #edt value
        
        #the point on the skeleton: 
        x = inds[0][white_spots_no_tail[i][0]][white_spots_no_tail[i][1]]
        y= inds[1][white_spots_no_tail[i][0]][white_spots_no_tail[i][1]]
        
        #*KEY: distance from the head, new edt(distance from skeleton)  *VALUE: spot's group
        #help_spots[pixels_dist_from_head_copy[x,y],new_edt] = spots_group[white_spots_no_tail[i][0],white_spots_no_tail[i][1]]
        try:
            help_spots[ pixels_dist_from_head_copy[x,y],new_edt] =spots_group[white_spots_no_tail[i][0],white_spots_no_tail[i][1]]
        except:
            continue;
        #                                      # X-distance from head        Y-distance from skeleton
        new_img = np.append(new_img, np.array([[ pixels_dist_from_head_copy[x,y],new_edt]]), axis=0)  
       
        spots_old_new[white_spots_no_tail[i][0],white_spots_no_tail[i][1]] = [pixels_dist_from_head_copy[x,y],new_edt]
        
        spots_old_new[white_spots_no_tail[i][0],white_spots_no_tail[i][1]] = [pixels_dist_from_head_copy[x,y],new_edt]
    
        
    #changing all the new image values to non negative - adding the largest negative number 
    min_value = min(np.min(new_img[:,1]),np.min(new_img_morph[:,1]))
    min_new_img = min_value
    new_img[:,1] -= min_value
    new_img = np.round(new_img)
    
    def get_key(val):
        for key, value in spots_old_new.items():
              if val == value:
                  return key
     
        return "key doesn't exist"
    
    # for v in spots_old_new.values():
    for k, v in spots_old_new.items():
        knew=list(v)
        # knew[1]-=min_value
        knew[1]=np.round(knew[1])
        knew[0]=np.round(knew[0])
        knew=np.array(knew).astype(np.int64)
        spots_old_new[k] = tuple(knew)
            
    for k in help_spots.keys():
        knew=list(k)
        knew[1]-=min_value
        knew[1]=np.round(knew[1])
        knew[0]=np.round(knew[0])
        knew=np.array(knew).astype(np.int64)
        #KEY: new x,y of the spot  VALUE- spot's group
        spots_new_img[tuple(knew)] = help_spots[k]
        
    def return_spots_old_new ():
        return spots_old_new
    ####################MORPH####################################3
    #changing all the new image values to non negative - adding the largest negative number 
    # min_new_img_morph = np.min(new_img_morph[:,1])
    new_img_morph[:,1] -= min_value
    new_img_morph = np.round(new_img_morph)
    
    
    
    # for v in spots_old_new.values():
    for k, v in morph_old_new.items():
        knew_morph=list(v)
        # knew_morph[1]-=min_value
        knew_morph[1]=np.round(knew_morph[1])
        knew_morph[0]=np.round(knew_morph[0])
        knew_morph=np.array(knew_morph).astype(np.int64)
        morph_old_new[k] = tuple(knew_morph)
    
    print("end morph")
    
    #######################
    #setting the size of the new images
    max_x = int(max(max(new_img_morph[:,0]),max(new_img[:,0])))
    max_y = int(max(max(new_img_morph[:,1]),max(new_img[:,1])))
    ###################
    
    
    new_img_morph= new_img_morph.astype(np.int64)
    # new_salamander =np.zeros(max(new_img[:,0]),max(new_img[:,1]))
    new_salamander_morph =np.empty((max_x +1,max_y+1), int)
    new_salamander_morph.fill(0)
    
    new_salamander_morph =np.zeros((max_x +1,max_y +1, 3))
    
    
    for i in range(new_img_morph.shape[0]):
        new_salamander_morph[new_img_morph[i][0],new_img_morph[i][1]] = 255
        
    filename = path+'\MORPH.jpg'
    cv2.imwrite(filename, new_salamander_morph) 
     
    print("image? morph")
    ###########################################################################################################################################33
    #creating new image:
    
    color_list=np.round(np.random.rand(len(np.unique(spots)+1),3)*255).astype(np.int64) #randomized color list of 49 colors
    
    # coloring the spots image 
    spots_copy =np.zeros((img.shape[0],img.shape[1],3))
    
    
    for i in range(white_spots_no_tail.shape[0]):
        color =spots_group[(white_spots_no_tail[i][0] ,white_spots_no_tail[i][1])]
        spots_copy[white_spots_no_tail[i][0] ,white_spots_no_tail[i][1],:] =color_list[color]
        
    new_img= new_img.astype(np.int64)
    # new_salamander =np.zeros(max(new_img[:,0]),max(new_img[:,1]))
    new_salamander =np.empty((max_x +1,max_y+1), int)
    new_salamander.fill(0)
    
    colored_salamander =np.zeros((max_x +1,max_y+1, 3))
    
    
    for i in range(new_img.shape[0]):
        new_salamander[new_img[i][0],new_img[i][1]] = 255
        color =spots_new_img[(new_img[i][0],new_img[i][1])]
        colored_salamander[new_img[i][0],new_img[i][1],:] =color_list[color]
    # ####################################
    new_min =round(-min_new_img)
    new_salamnder_with_skeleton =np.copy(colored_salamander)
    for i in range(new_salamnder_with_skeleton.shape[0]):
        new_salamnder_with_skeleton[i,new_min,:] =255
    #####################
    change_white =np.copy(white)
    colored_salamander_with_skelton =np.copy(spots_copy)
    for i in range(white.shape[0]):
            colored_salamander_with_skelton[white[i][0],white[i][1],:] =255
    
    
    filename = path+'\A_COLORED_SALAMANDER.jpg'
    cv2.imwrite(filename, colored_salamander)      
    filename2 = path+'\A_SALAMANDER_blackandwhite.jpg'
    cv2.imwrite(filename2, new_salamander)
    filename3 = path+'\A_ORIGINAL_COLORED_SALAMANDER_new_skeleton.jpg'
    cv2.imwrite(filename3, spots_copy) 
    
    filename4 =path+'\A_colored_salamander_with_skelton.jpg'
    cv2.imwrite(filename4, colored_salamander_with_skelton)
    
    filename5 =path+'\A_NEW_with_skelton0.3.jpg'
    cv2.imwrite(filename5, new_salamnder_with_skeleton)
    
    ##################################################################################################3
    #morphology for fill the new spots 
    
    #setting new kernel for morphology
    kernel = np.ones((kernel_par,kernel_par),np.uint8)
    #new black image for the morphology 
    copy_new_salamander =np.zeros((new_salamander.shape[0],new_salamander.shape[1]))
    #the black and white image of the spots after morphology 
    copy_new_salamander_2 =np.zeros((new_salamander.shape[0],new_salamander.shape[1]))
    
    #tcolored image of the spots after morphology 
    copy_new_salamander_3  =np.zeros((new_salamander.shape[0],new_salamander.shape[1],3))
    
    
    #dict of the new x,y after morphology and the spots num (color)
    spots_after_mophology ={}
    #itrate over the spots - from label 1 to label 48 
    for i in range(1 ,len(np.unique(spots))):
        for k in spots_new_img.keys(): # itrate over the spots indexes
            if(spots_new_img[k] ==i):# if the value of the x, y of the spot equals to the spot number we color it to white in the new image 
                copy_new_salamander[[k[0]],[k[1]]]=255
        #morphology to a spot
        new_image_morph  = cv2.morphologyEx(copy_new_salamander, cv2.MORPH_CLOSE, kernel)
        
        #saving the white indexes of the new spot
        white_morph =np.argwhere(new_image_morph == 255)
        
    #new dictionary - saving the spots label for all x,y after the morphology 
        for j in range(white_morph.shape[0]):
            spots_after_mophology[white_morph[j][0],white_morph[j][1]] =i
            
    #Join the new spot to the new after morpholgy spot image- or between 2 images- the one we save and the other of new spot after morphology      
        copy_new_salamander_2 = cv2.bitwise_or(copy_new_salamander_2,new_image_morph)
        copy_new_salamander =np.zeros((new_salamander.shape[0],new_salamander.shape[1]))
        
    # all the spots after morphology    
    white_sopts_after_morph =np.argwhere(copy_new_salamander_2 == 255)
    #coloring the after morphology image -> with the spots_after_mophology dictionary
    for i in range(white_sopts_after_morph.shape[0]):
        color =spots_after_mophology[(white_sopts_after_morph[i][0],white_sopts_after_morph[i][1])]
        copy_new_salamander_3[white_sopts_after_morph[i][0],white_sopts_after_morph[i][1],:] =color_list[color]
    
    white_spots_with_skeleton = np.copy(copy_new_salamander_2)
    new_salamnder_morphology_with_skeleton =np.copy(copy_new_salamander_3)
    for i in range(new_salamnder_morphology_with_skeleton.shape[0]):
        new_salamnder_morphology_with_skeleton[i,new_min,:] =255
        white_spots_with_skeleton[i,new_min] =255
    
    filename6 =path+'\A_MORPHOLOGY__COLOR-ker'+str(kernel_par)+'.png'
    cv2.imwrite(filename6, new_salamnder_morphology_with_skeleton)
    
    
    filename7 =path+'\black_white_MORPHOLOGY__COLOR-ker'+str(kernel_par)+'.png'
    cv2.imwrite(filename7, copy_new_salamander_2)
    
    filename8 =path+'\black_white_Skeleton_MORPHOLOGY__COLOR-ker'+str(kernel_par)+'.png'
    cv2.imwrite(filename8, white_spots_with_skeleton)
    
    return spots_old_new, morph_old_new
    
#sal1= pickle.load( open( "AH181965_1.p", "rb" ) )
# sal1 = np.copy(sal11)
#spots_old, morph_old_new =  run_svd( sal1.final_skel, sal1.spots_old,sal1.morphology,5 ,sal1.head, sal1.tail,sal1.path)

# run_svd(img, spots_img, morph_img, 5,head, tail,path)

