# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 20:06:25 2022

@author: user
"""
import numpy as np
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
import cv2
from skimage import measure
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from new_skeleton import create_skeleton
from remove_legs_new import remove_legs_from_skeleton
import connectcomponents as connect
from find_head import head_tail
import graph_svd_new as svd
#import graph_svd_old as svd
import spots_head as head 
import numpy as np
import pickle
import remove_spots
import math

class Spots:
                
    def __init__(self, img, old_spots): 
        self.img_spots = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        self.blobs = create_blobs(self.img_spots)
        self.spots_group ={}
        self.spots_group = only_spots(self.img_spots ,self.spots_group, self.blobs)
        self.num_spots = len(np.unique(self.blobs))
        self.blobs_spots ={}
        self.blobs_spots =  create_spots_blobs(self.blobs_spots ,self.num_spots, self.spots_group)
        self.spots = []
        self.spots_dic = {}
        # self.spots = new_spot(self)
        # self.spots_dic = new_spot(self)
        self.spots, self.spots_dic =new_spot(self.spots, self.spots_dic , self.blobs_spots,old_spots)

                
def new_spot(spots,spots_dic,blobs_spots,old_spots):
        for k in blobs_spots.keys():
            new_spot = Spot(blobs_spots.get(k),k,old_spots) 
            spots.append(new_spot)
            spots_dic[k] =new_spot
        return spots,spots_dic
        
class Spot:
    def __init__(self, spot, num,old_spots): 
        self.name = "spot"+str(num)
        self.num =num
        self.new_spot =np.array(spot) 
        self.pixels_num = self.new_spot[0].shape[0]
        # print(num)
        max_values = np.amax(self.new_spot, axis=1)
        # print("--------------------")
        # print(max_values)
        self.x_max = max_values[0][0]
        # print( self.x_max)
        self.y_max =  max_values[0][1]
        min_values = np.amin( self.new_spot, axis=1)
        self.x_min =min_values[0][0]
        self.y_min =min_values[0][1]
        self.center_x =int((self.x_max + self.x_min )/2)
        self.center_y = int((self.y_max +self.y_min )/2)
        #
        try:
            self.norm_x = old_spots[self.center_x,self.center_y][0]
            self.norm_y = old_spots[self.center_x,self.center_y][1]
        except:
            print("except--------------------")
            print(self.num)
            print(self.center_x)
            print(self.center_y)
            print("end of except --------------------")
            self.norm_x = 9999
            self.norm_y = 9999
        #
        # cv2.circle(img_spots,(self.center_y,  self.center_x ), 3, (0,255,0), -1)
        
        self.rectangle_area =(self.x_max -self.x_min)*(self.y_max -self.y_min)
        self.percentange_rec = (self.pixels_num / self.rectangle_area)*100
        
        # cv2.circle(new_salamander_morph,(self.norm_y,  self.norm_x ), 3, (0,255,0), -1)
        # cv2.circle(colored_salamander,(self.norm_y,  self.norm_x ), 3, (0,255,0), -1)
        
        # filename ='newwwwwwwwwwww.jpg'
        # cv2.imwrite(filename, new_salamander_morph)
        
        # filename ='newwwwwwwwwwww2.jpg'
        # cv2.imwrite(filename, colored_salamander)
        
        # filename ='newwwwwwwwwwww3.jpg'
        # cv2.imwrite(filename, img_spots)

        #############################
        #       SVD
        mean=np.mean(self.new_spot[0], axis=0)
        self.new_spot[0][:,0] =self.new_spot[0][:,0] -mean[0]
        self.new_spot[0][:,1] =self.new_spot[0][:,1] -mean[1]
        
        U, sigma, V = np.linalg.svd(self.new_spot[0] , full_matrices=False)
        self.v1 = V[0,:]
        self.v2 = V[1,:]
        self.sigma1 =sigma[0]
        self.sigma2 =sigma[1]
        self.sigma_ratio = self.sigma2/self.sigma1
        self.vac = np.array([self.pixels_num,self.norm_x,self.norm_y,self.rectangle_area,           
                    self.percentange_rec, self.sigma1,self.sigma2, self.sigma_ratio])
#################
#  x, y -> normelized
# vector 



       

def create_blobs(img):
    blobs = img
    all_labels = measure.label(blobs)
    blobs_labels = measure.label(blobs, background=0)
    return(blobs_labels)


                    
def only_spots(img_spots, spots_group, blobs):
        only_spots = np.empty((0,2), int)
        for x in range(img_spots.shape[0]):
            for y in range(img_spots.shape[1]):
                if(img_spots[x,y]!=0):
                    only_spots = np.append(only_spots, np.array([[x,y]]), axis=0)
        for i in range(only_spots.shape[0]):
            spots_group[only_spots[i][0],only_spots[i][1]] =blobs[only_spots[i][0]][only_spots[i][1]] 
        return spots_group
    
def create_spots_blobs(blobs_spots,num_spots,spots_group):
            for i in range(1,num_spots):
                blobs_spots[i] =[]

            for k in blobs_spots.keys():
                blobs_spots[k].append([key for key, v in spots_group.items() if v == k])
                #new spot
                
            return blobs_spots

def calc_dis(p1,p2):
    distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
    return distance

def nearest_spot (s):
    nearest_spot
    sopt_point ={} # spots num value- x,y 
    points ={}
    all_points =[]
    distance_list =[]
    near_spots ={}
    all_near_spots =[]
    for i in range(len(s.spots.spots)):
        sopt_point[s.spots.spots[i].num] =[s.spots.spots[i].center_x,s.spots.spots[i].center_y]
        points[s.spots.spots[i].center_x,s.spots.spots[i].center_y] =s.spots.spots[i].num
        all_points.append([s.spots.spots[i].center_x,s.spots.spots[i].center_y])
    for i in range(len(all_points)):
        p2_dis ={}
        dis_p2 ={}
        distance_list=[]
        for j in range(len(all_points)):
            if(i!=j):
                p1 =all_points[i]
                p2=all_points[j]
                distance = calc_dis(p1,p2)
                p2_dis[points[(p2[0],p2[1])]] =distance
                dis_p2[distance] =points[(p2[0],p2[1])]
                distance_list.append(distance)
    
        distance_list.sort() 
        l=[]
        for k in range (0,3):
            # v ={i for i in p2_dis if p2_dis[i]==distance_list[k]}
            # dis_p2[k]
            # print(dis_p2[k])
            l.append((dis_p2[distance_list[k]]))
        # for k in range(1,4):
        #     v ={i for i in p2_dis if p2_dis[i]==distance_list[k]}
        #     s = list(v)
        #     print(s)
        #     l.append(s)
        
        near_spots[points[(p1[0],p1[1])]]   =  l#distance_list
        all_near_spots.append(near_spots)
    return near_spots

class salamander():
    #
    #
    def __init__(self, original_img ,body_img, spots_img,path, name ):
        self.name = name
        self.original_img = original_img     
        self.body_img = body_img
        
        self.path = path
        
        # filename = r'C:\Users\user\OneDrive - University of Haifa\Desktop\Project\class_salamander\try\FROMֹTHEֹCLASSbody_img.jpg'
        # cv2.imwrite(filename, self.body_img )
        #salamader skeleton - with legs and tail 
        #morphology - closing holes in the bogy_img 
        
        #                                      kernel parameter, usually 15
        self.skeleton,self.morphology = create_skeleton(self.body_img , 25,self.path)
        
        
        #removing legs and tail from the skeleton
        # self.less_25 =remove_legs_from_skeleton(self.morphology , self.skeleton ,25,self.path)
        #longer skeleton in order to find the head later
        self.less_30 =remove_legs_from_skeleton(self.morphology , self.skeleton ,30,self.path)
          #connect ->  30 and 25 
        self.connected_skel = connect.connected_skleton(self.less_30,self.path)
        # self.connected_skel_long = connect.connected_skleton(self.less_25)
        #clean skeleton 
        self.final_skel = connect.clean_skeleton(self.connected_skel,self.path)
        self.spots_img =remove_spots.remove_spots(self.final_skel, spots_img,self.path )
        #find head and tail 
        self.head, self.tail = head.return_head_tail(self.final_skel, self.spots_img)
        # head_tail(self.connected_skel_long,self.connected_skel)
        #svd                                                         
        self.spots_old_new, self.body_old_new = svd.run_svd( self.final_skel, self.spots_img,self.morphology,5 ,self.head, self.tail,self.path)
        #spots 
        self.spots = Spots(self.spots_img,self.body_old_new)
        self.nearest_neighbors =nearest_spot (self)