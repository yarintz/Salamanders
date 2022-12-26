# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:05:16 2022

@author: user
"""
import pandas as pd
import numpy as np
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
import cv2
import numpy as np
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
import pickle
import random 
import math
import cv2
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

all_near_spots =[]
near_spots ={}


list_names =["sal1","num1","pixels_num1", "norm_x1","norm_y1","rectangle_area1",           
                    "percentange_rec1", "sigma1_1","sigma2_1", "sigma_ratio1",
                   
                    #
                    # "pixels_num1_1", "norm_x1_1","norm_y1_1","rectangle_area1_1",           
                    # "percentange_rec1_1", "sigma1_1_1","sigma2_1_1", "sigma_ratio1_1",
                    # #
                    # "pixels_num1_2", "norm_x1_2","norm_y1_2","rectangle_area1_2",           
                    # "percentange_rec1_2", "sigma1_1_2","sigma2_1_2", "sigma_ratio1_2",
                    # #
                    # "pixels_num1_3", "norm_x1_3","norm_y1_3","rectangle_area1_3",           
                    # "percentange_rec1_3", "sigma1_1_3","sigma2_1_3", "sigma_ratio1_3",
                    #
                    "scale",
                    "sal2","num2","pixels_num2", "norm_x2","norm_y2","rectangle_area2",           
                    "percentange_rec2", "sigma1_2","sigma2_2", "sigma_ratio2",
                    # #2_1
                    # "pixels_num2_1", "norm_x2_1","norm_y2_1","rectangle_area2_1",           
                    # "percentange_rec2_1", "sigma1_2_1","sigma2_2_1", "sigma_ratio2_1",
                    # #2_2
                    # "pixels_num2_2", "norm_x2_2","norm_y2_2","rectangle_area2_2",           
                    # "percentange_rec2_2", "sigma1_2_2","sigma2_2_2", "sigma_ratio2_2",
                    # #2_3
                    # "pixels_num2_3", "norm_x2_3","norm_y2_3","rectangle_area2_3",           
                    # "percentange_rec2_3", "sigma1_2_3","sigma2_2_3", "sigma_ratio2_3",
                    # #
                    "pixels_num3", "norm_x3","norm_y3","rectangle_area3",           
                    "percentange_rec3", "sigma1_3","sigma2_3", "sigma_ratio3",
                    
                    "y"
                    # "num","pixels_num", "norm_x","norm_y","rectangle_area",           
                    # "percentange_rec", "sigma1","sigma2", "sigma_ratio"
                    ]
# d =pickle.load( open( "AH1819116_spots.p", "rb" ) )

# m =s1.morphology
# morph =np.argwhere(m == 255)
# d3 = s1.spots.spots_dic
def return_head_tail(body):
    # after_skel = cv2.imread('After_morphology.jpg', 0)
    body = cv2.threshold(body, 127, 255, cv2.THRESH_BINARY)[1]
    only_spots = np.argwhere(body == 255)
    blobs = body
    all_labels = measure.label(blobs)
    blobs_labels = measure.label(blobs, background=0)
    coor_and_labels ={}#coordinates and their spot's label

    for i in range(only_spots.shape[0]):
         coor_and_labels[only_spots[i][0], only_spots[i][1]] =  blobs_labels[only_spots[i][0]][only_spots[i][1]]
    
    max_pixels = 0;
    for i in range (len(np.unique(blobs_labels))):
        label_1 = [k for k, v in coor_and_labels.items() if v == i]
        # print(str(i)+" "+str(len(label_1)))
    
       
       
       
        if(len(label_1)>max_pixels):
            max_pixels = len(label_1)
            biggest_spot = label_1
    
    # print(len(biggest_spot))
            
    return len(biggest_spot)

def calc_dis(p1,p2):
    distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
    return distance
# def new_svd(spot):
#     mean=np.mean(spot.new_spot[0], axis=0)
#     spot.new_spot[0][:,0] =spot.new_spot[0][:,0] -mean[0]
#     spot.new_spot[0][:,1] =spot.new_spot[0][:,1] -mean[1]
            
#     U, sigma, V = np.linalg.svd(spot.new_spot[0] , full_matrices=False)
#     spot.v1 = V[0,:]
#     spot.v2 = V[1,:]
#     spot.sigma1 =sigma[0]
#     print("_________________")
#     print(spot.num)
#     # print("sigma")
#     # print( spot.sigma1)
    
#     spot.sigma2 =sigma[1]
#     # print( spot.sigma2)
#     print("vector")
                       
#     spot.sigma_ratio = spot.sigma2/spot.sigma1
#     spot.vac = np.array([spot.pixels_num,spot.norm_x,spot.norm_y,spot.rectangle_area,           
#                     spot.percentange_rec, spot.sigma1,spot.sigma2, spot.sigma_ratio])
#     print("function")
#     print(spot.vac[5])
def new_svd(s):
    for i in range(len(s.spots.spots)):
        mean=np.mean(s.spots.spots[i].new_spot[0], axis=0)
        s.spots.spots[i].new_spot[0][:,0] =s.spots.spots[i].new_spot[0][:,0] -mean[0]
        s.spots.spots[i].new_spot[0][:,1] =s.spots.spots[i].new_spot[0][:,1] -mean[1]
                
        U, sigma, V = np.linalg.svd(s.spots.spots[i].new_spot[0] , full_matrices=False)
        s.spots.spots[i].v1 = V[0,:]
        s.spots.spots[i].v2 = V[1,:]
        s.spots.spots[i].sigma1 =sigma[0]
        print("_________________")
        print(s.spots.spots[i].num)
        # print("sigma")
        # print( spot.sigma1)
        
        s.spots.spots[i].sigma2 =sigma[1]
        # print( spot.sigma2)
        print("vector")
                           
        s.spots.spots[i].sigma_ratio = s.spots.spots[i].sigma2/s.spots.spots[i].sigma1
        # s.spots.spots[i].vac = np.array([s.spots.spots[i].pixels_num,s.spots.spots[i].norm_x,
        #                                    s.spots.spots[i].norm_y,s.spots.spots[i].rectangle_area,           
        #                 s.spots.spots[i].percentange_rec,
        #                 s.spots.spots[i].sigma1,s.spots.spots[i].sigma2, s.spots.spots[i].sigma_ratio])
        s.spots.spots[i].vac[5] =s.spots.spots[i].sigma1
        s.spots.spots[i].vac[6] =s.spots.spots[i].sigma2
        s.spots.spots[i].vac[7] =s.spots.spots[i].sigma_ratio
        
        print("function")
        print(s.spots.spots[i].vac[5])

    
def nearest_spot (s):
    nearest_spot
    sopt_point ={} # spots num value- x,y 
    points ={}
    all_points =[]
    distance_list =[]
    near_spots ={}
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
        
    
     
    
def calc_scale(s1, s2):
    morph1 = s1.morphology
    morph2 = s2.morphology
    scale =(return_head_tail(morph1)/return_head_tail(morph2))**.5
   # scale =(len(np.argwhere(morph1 == 255))/len(np.argwhere(morph2 == 255)))**.5
    if(scale>1):
        return scale, 2
    else:
        return scale, 2 
def is_zero(v1,v2):
    if(v1==0 or v2==0):
        return 0
    else:
        return v1/v2
   
def change_vectors(v, scale):
    # v[0] = v[0]/(scale**2)
    # v[1] = np.round(v[1]/scale)
    # v[2] = np.round(v[2]/scale)
    # v[3] = v[3]/(scale**2)
    # v[4] = v[4]#/(scale**2)
    # v[7] = ( v[6]/v[5])/scale
    # v[7] = (is_zero(v[6],v[5])/scale)
    # v[5] = v[5]/scale
    # v[6] = v[6]/scale
    v[0] = v[0]*(scale**2)
    v[1] = np.round(v[1]*scale)
    v[2] = np.round(v[2]*scale)
    v[3] = v[3]*(scale**2)
    v[4] = v[4]#/(scale**2)
    v[7] = ( v[6]/v[5])*scale
    v[7] = (is_zero(v[6],v[5])*scale)
    v[5] = v[5]*scale
    v[6] = v[6]*scale

    return v

    
def new_vactor (v1,v2,scale,scale_norm_x,scale_norm_y):
    # v3 = np.array([v1[0]/v2[0], v1[1]/v2[1],v1[2]/v2[2],v1[3]/v2[3]
    #                ,v1[4]/v2[4],v1[5]/v2[5],v1[6]/v2[6],(v1[6]/v1[5])/(v2[6]/v2[5])])
    v3 = np.array([is_zero(v1[0],v2[0]*(scale**2)),
                   is_zero(v1[1],np.round(v2[1]*scale_norm_x)),
                   is_zero(v1[2],np.round(v2[2]*scale_norm_y)),
                   is_zero(v1[3],(v2[3]*(scale**2))),
                   is_zero(v1[4],v2[4]),
                   is_zero(v1[5],v2[5]*scale),
                   is_zero(v1[6],v2[6]*scale),
                  is_zero( is_zero(v1[6],v1[5]), is_zero(v2[6],v2[5]))
                   ])
    return v3
    
        

def create_df(s1,s2,list_a_b,name1,name2):

    df = pd.DataFrame(columns=list_names)
    # near1 =nearest_spot(s1)
    # near2 =nearest_spot(s2)

    spots_num_a = [*range(1, len(s1.spots.spots)+1, 1)]
    spots_num_b = [*range(1, len(s2.spots.spots)+1, 1)]
    
    pairs = [[w,f] for w in spots_num_a for f in spots_num_b]
    all_spots_list = list.copy(list_a_b)
    # pairs = set(pairs).symmetric_difference(list_a_b)
    for i in range (len(list_a_b)*2):
        # choice = [random.choice(pairs) for _ in range (1)]
        # if([choice] in list_a_b):
        #     choice = [random.choice(pairs) for _ in range (1)]
        # all_spots_list.append(choice[0])
        choice = [random.choice(pairs) for _ in range (1)]
        while([choice] in list_a_b):
                choice = [random.choice(pairs) for _ in range (1)]
        # if not([choice] in list_a_b):   
        all_spots_list.append(choice[0])
        
    
    random.shuffle(all_spots_list)  
    y = 0
    scale, choosen_sal = calc_scale(s1, s2)
    # vectors_dict ={}
    # for i in range(len(s2.spots.spots)):
    #     vectors_dict[i] = change_vectors(s2.spots.spots[i-1].vac, scale)
        
    for i in range (len(all_spots_list)):
        # print(all_spots_list[1][0])
        a = all_spots_list[i][0]
        b = all_spots_list[i][1]
        ########
        #SPOT 1
        
        # print("BEFORE")
        # print(s1.spots.spots[a-1].vac[5])
        # new_svd(s1.spots.spots[a-1]) #change sigma values
        # new_svd(s1,a-1)
        v1 =s1.spots.spots[a-1].vac
        # print("NOTTT function")
        # print(v1[5])
        # num1 =s1.spots.spots[a-1].num
        
        # SPOT 2
       
        # new_svd(s2.spots.spots[b-1]) #change sigma values
        # new_svd(s2,b-1)
        # v2 =s2.spots.spots[b-1].vac
        #v2 =s2.spots.spots[b-1].vac

        v2 =s2.spots.spots[b-1].vac
        # vectors_dict[b-1]
        # change_vectors(s2.spots.spots[b-1].vac, scale)
        
        # num2 =s2.spots.spots[b-1].num
        ######
        # near_a =near1[num1]
        # print("near_a")
        # print(near_a)
        # v1_1 =s1.spots.spots[near_a[0]-1].vac
        # v1_2 =s1.spots.spots[near_a[1]-1].vac
        # v1_3 =s1.spots.spots[near_a[2]-1].vac
        # near_b =near2[num2]
        # print("_______________________")
        # print("near_b")
        # print(near_b)
        # v2_1 =s2.spots.spots[near_b[0]-1].vac
        # v2_2 =s2.spots.spots[near_b[1]-1].vac
        # v2_3 =s2.spots.spots[near_b[2]-1].vac

        
        # print(" B")
        # print(b)
        # print(" A")
        # print(a)
        # v1 =s1.spots.spots[a-1].vac
        # # print(v1)
        # v2 =s2.spots.spots[b-1].vac
        # if(choosen_sal == 1):
        #     v1 =change_vectors(v1, scale)
        # else:
        #     v2 =change_vectors(v2, scale)
        #     # v2_1 =change_vectors(v2_1, scale)
        #     # v2_2 =change_vectors(v2_2, scale)
        #     # v2_3 =change_vectors(v2_3, scale)
        
        # max_x_1= max (spot.norm_x for spot in s1.spots.spots)
        # max_y_1= max (spot.norm_y for spot in s1.spots.spots)
        # max_x_2= max (spot.norm_x for spot in s2.spots.spots)
        # max_y_2= max (spot.norm_y for spot in s2.spots.spots)
        # scale_norm_x = max_x_1/max_x_2
        # scale_norm_y = max_y_1/max_y_2
        max_x_1= max (s1.body_old_new[spot][0] for spot in s1.body_old_new)
        max_y_1= max (s1.body_old_new[spot][1] for spot in s1.body_old_new)
        max_x_2= max (s2.body_old_new[spot][0] for spot in s2.body_old_new)
        max_y_2= max (s2.body_old_new[spot][1] for spot in s2.body_old_new)
        scale_norm_x = max_x_1/max_x_2
        scale_norm_y = max_y_1/max_y_2
        v3 = new_vactor (v1,v2,scale,scale_norm_x,scale_norm_y)
            
            
    
        if([a,b] in list_a_b):
            y = 1
            # print(y)
        else:
            y=0
        df = df.append({'sal1':name1 ,'num1':a, 'pixels_num1': v1[0], 'norm_x1': v1[1],
                'norm_y1': v1[2],'rectangle_area1': v1[3],'percentange_rec1': v1[4],
                'sigma1_1':v1[5],
                'sigma2_1' : v1[6],'sigma_ratio1': is_zero(v1[6],v1[5]),
                # #1_1
                # # 'num1_1':near_a[0], 
                # 'pixels_num1_1': v1_1[0], 'norm_x1_1': v1_1[1],
                # 'norm_y1_1': v1_1[2],'rectangle_area1_1': v1_1[3],'percentange_rec1_1': v1_1[4],
                # 'sigma1_1_1':v1_1[5],
                # 'sigma2_1_1' : v1_1[6],'sigma_ratio1_1': is_zero(v1_1[6],v1_1[5]),
                # #1_2
                #   # 'num1_2':near_a[1],
                #   'pixels_num1_2': v1_2[0], 'norm_x1_2': v1_2[1],
                # 'norm_y1_2': v1_2[2],'rectangle_area1_2': v1_2[3],'percentange_rec1_2': v1_2[4],
                # 'sigma1_1_2':v1_2[5],
                # 'sigma2_1_2' : v1_2[6],'sigma_ratio1_2': is_zero(v1_2[6],v1_2[5]),
                # #1_3
                # # 'num1_3':near_a[2], 
                # 'pixels_num1_3': v1_3[0], 'norm_x1_3': v1_3[1],
                # 'norm_y1_3': v1_3[2],'rectangle_area1_3': v1_3[3],'percentange_rec1_3': v1_3[4],
                # 'sigma1_1_3':v1_3[5],
                # 'sigma2_1_3' : v1_3[6],'sigma_ratio1_3': is_zero(v1_3[6],v1_3[5]),

                
                 'scale':scale,
                 'sal2':name2,
                 'num2': b, 
                 'pixels_num2': v2[0]*(scale**2),
                 'norm_x2': np.round(v2[1]*scale_norm_x),
                'norm_y2': np.round(v2[2]*scale_norm_y),
                'rectangle_area2': v2[3]*(scale**2),
                'percentange_rec2':v2[4],
                'sigma1_2': v2[5]*scale,
                'sigma2_2': v2[6]*scale,
                'sigma_ratio2': is_zero(v2[6],v2[5]),
                # #2_1
                # # 'num2_1':near_b[0],
                # 'pixels_num2_1': v2_1[0], 'norm_x2_1': v2_1[1],
                # 'norm_y2_1': v2_1[2],'rectangle_area2_1': v2_1[3],'percentange_rec2_1': v2_1[4],
                # 'sigma1_2_1':v2_1[5],
                # 'sigma2_2_1' : v2_1[6],'sigma_ratio2_1': is_zero(v2_1[6],v2_1[5]),
                # #2_2
                #   # 'num2_2':near_b[1],
                #   'pixels_num2_2': v2_2[0], 'norm_x2_2': v2_2[1],
                # 'norm_y2_2': v2_2[2],'rectangle_area2_2': v2_2[3],'percentange_rec2_2': v2_2[4],
                # 'sigma1_2_2':v2_2[5],
                # 'sigma2_2_2' : v2_2[6],'sigma_ratio2_2': is_zero(v2_2[6],v2_2[5]),
                # #2_3
                # # 'num2_3':near_b[2],
                # 'pixels_num2_3': v2_3[0], 'norm_x2_3': v2_3[1],
                # 'norm_y2_3': v2_3[2],'rectangle_area2_3': v2_3[3],'percentange_rec2_3': v2_3[4],
                # 'sigma1_2_3':v2_3[5],
                # 'sigma2_2_3' : v2_3[6],'sigma_ratio2_3': is_zero(v2_3[6],v2_3[5]),
                #
                'pixels_num3': v3[0],
                'norm_x3': v3[1],
                'norm_y3': v3[2],
                'rectangle_area3': v3[3],
                'percentange_rec3':v3[4],
                'sigma1_3': v3[5],
                'sigma2_3': v3[6],
                'sigma_ratio3': v3[7],
                "y":y} , ignore_index = True) 
    # vectors_dict.clear()
    return df
 ##########################
##1
#AH1819116    
# AH1819116_2 = pickle.load( open( "AH1819116_2.p", "rb" ) )
# AH1819116_3 = pickle.load( open( "AH1819116_3.p", "rb" ) )
# AH1819116_4 = pickle.load( open( "AH1819116_4.p", "rb" ) )
# new_svd(AH1819116_2)
# new_svd(AH1819116_3)
# new_svd(AH1819116_4)
# AH1819116_dict = {}
# AH1819116_dict[2] = AH1819116_2
# AH1819116_dict[3] = AH1819116_3
# AH1819116_dict[4] = AH1819116_4
# s="AH1819116"
# # list_a_b =pickle.load( open( "AH1819116_2_AH1819116_3.p", "rb" ) )    
# # df = create_df(AH1819116_2,AH1819116_2,list_a_b)

# df_AH1819116 ={}
# for i in range(2,5):
#     for j in range(3,5):
#         print("-----------------------")
#         print(i)
#         print(j)
#         if(i!=j and i<j):
#             pickfilename=s+ r"_"+ str(i)+ r"_"+ s + r"_"+ str(j)
#             print(pickfilename)
#             list_a_b =pickle.load( open( pickfilename +".p", "rb" ) ) 
#             df1 =create_df(AH1819116_dict[i],AH1819116_dict[j],list_a_b,s+ r"_"+ str(i),s + r"_"+ str(j))
#             df_AH1819116[str(i)+"_"+str(j)] =df1
                
# df_AH1819116.values()    
# df_merged = pd.concat([df_AH1819116["2_3"],
#                         df_AH1819116["2_4"],
#                         df_AH1819116["3_4"]], ignore_index=True, sort=False)
# all_sal_df = pickle.load( open( "all_sal_df.p", "rb" ) )
# all_sal_df["AH1819116"] =df_merged
# pickle.dump( all_sal_df, open("all_sal_df.p", "wb" ) ) 
# pickle.dump( df_merged, open("AH1819116_df.p", "wb" ) )  
# # ####################
# # ##2
# # #AH181983
# AH181983_2 = pickle.load( open( "AH181983_2.p", "rb" ) )
# AH181983_3 = pickle.load( open( "AH181983_3.p", "rb" ) )
# AH181983_4 = pickle.load( open( "AH181983_4.p", "rb" ) )
# s="AH181983"
# AH181983_dict = {}
# AH181983_dict[2] = AH181983_2
# AH181983_dict[3] = AH181983_3
# AH181983_dict[4] = AH181983_4
# new_svd(AH181983_2)
# new_svd(AH181983_3)
# new_svd(AH181983_4)

# df_AH181983 ={}
# for i in range(2,5):
#     for j in range(3,5):
#         print("-----------------------")
#         print(i)
#         print(j)
#         if(i!=j and i<j):
#             pickfilename=s+ r"_"+ str(i)+ r"_"+ s + r"_"+ str(j)
#             print(pickfilename)
#             list_a_b =pickle.load( open( pickfilename +".p", "rb" ) ) 
#             df1 =create_df(AH181983_dict[i],AH181983_dict[j],list_a_b, s+ r"_"+ str(i),s + r"_"+ str(j))
#             df_AH181983[str(i)+"_"+str(j)] =df1
                
# df_merged = pd.concat([df_AH181983["2_3"],
#                         df_AH181983["2_4"],
#                         df_AH181983["3_4"]], ignore_index=True, sort=False)
# all_sal_df = pickle.load( open( "all_sal_df.p", "rb" ) )
# all_sal_df["AH181983"] =df_merged
# pickle.dump( all_sal_df, open("all_sal_df.p", "wb" ) ) 
# pickle.dump( df_merged, open("AH181983_df.p", "wb" ) )  
# # # #######################
# # # ##3
# # #AH181967
# AH181967_1 = pickle.load( open( "AH181967_1.p", "rb" ) )
# AH181967_2 = pickle.load( open( "AH181967_2.p", "rb" ) )
# AH181967_3 = pickle.load( open( "AH181967_3.p", "rb" ) )
# AH181967_4 = pickle.load( open( "AH181967_4.p", "rb" ) )
# AH181967_5 = pickle.load( open( "AH181967_5.p", "rb" ) )
# AH181967_6 = pickle.load( open( "AH181967_6.p", "rb" ) )
# new_svd(AH181967_1)
# new_svd(AH181967_2)
# new_svd(AH181967_3)
# new_svd(AH181967_4)
# new_svd(AH181967_5)
# new_svd(AH181967_6)
# s="AH181967"
# AH181967_dict = {}
# AH181967_dict[1] = AH181967_1
# AH181967_dict[2] = AH181967_2
# AH181967_dict[3] = AH181967_3
# AH181967_dict[4] = AH181967_4
# AH181967_dict[5] = AH181967_5
# AH181967_dict[6] = AH181967_6
# df_AH181967 ={}
# for i in range(1,7):
#     for j in range(2,7):
#         print("-----------------------")
#         print(i)
#         print(j)
#         if(i!=j and i<j):
#             pickfilename=s+ r"_"+ str(i)+ r"_"+ s + r"_"+ str(j)
#             print(pickfilename)
#             list_a_b =pickle.load( open( pickfilename +".p", "rb" ) ) 
#             df1 =create_df(AH181967_dict[i],AH181967_dict[j],list_a_b, s+ r"_"+ str(i),s + r"_"+ str(j))
#             df_AH181967[str(i)+"_"+str(j)] =df1
                
# df_merged = pd.concat([df_AH181967["1_2"],
#                         df_AH181967["1_3"],
#                       df_AH181967["1_4"],
#                       df_AH181967["1_5"],
#                       df_AH181967["1_6"],
#                       df_AH181967["2_3"],
#                       df_AH181967["2_4"],
#                       df_AH181967["2_5"],
#                       df_AH181967["2_6"],
#                       df_AH181967["3_4"],
#                       df_AH181967["3_5"],
#                       df_AH181967["3_6"],
#                       df_AH181967["4_5"],
#                       df_AH181967["4_6"],
#                       df_AH181967["5_6"]
#                         ], ignore_index=True, sort=False)
# all_sal_df = pickle.load( open( "all_sal_df.p", "rb" ) )
# all_sal_df["AH181967"] =df_merged
# pickle.dump( all_sal_df, open("all_sal_df.p", "wb" ) ) 
# pickle.dump( df_merged, open("AH181967_df.p", "wb" ) )  
# ##############
# #4
# #AH171823
# AH171823_1 = pickle.load( open( "AH171823_1.p", "rb" ) )
# AH171823_2 = pickle.load( open( "AH171823_2.p", "rb" ) )
# AH171823_3 = pickle.load( open( "AH171823_3.p", "rb" ) )
# AH171823_5 = pickle.load( open( "AH171823_5.p", "rb" ) )
# new_svd(AH171823_1)
# new_svd(AH171823_2)
# new_svd(AH171823_3)
# new_svd(AH171823_5)
# s="AH171823"
# AH171823_dict = {}
# AH171823_dict[1] = AH171823_1
# AH171823_dict[2] = AH171823_2
# AH171823_dict[3] = AH171823_3
# AH171823_dict[5] = AH171823_5
# df_AH171823 ={}
# df_AH171823__NEWWW={}

# list_1 =[1,2,3,5]
# list_2 = [2,3,5]
# #NEED TO CHANGE

# # list_1 =[3]
# # list_2 = [5]

# for i in list_1:
#     for j in list_2:
#         # print("-----------------------")
#         # print(i)
#         # print(j)
#         if(i!=j and i<j):
#             pickfilename=s+ r"_"+ str(i)+ r"_"+ s + r"_"+ str(j)
#             print(pickfilename)
#             list_a_b =pickle.load( open( pickfilename +".p", "rb" ) ) 
#             df1 =create_df(AH171823_dict[i],AH171823_dict[j],list_a_b, s+ r"_"+ str(i),s + r"_"+ str(j))
#             # df_AH171823__NEWWW[str(i)+"_"+str(j)] =df1
#             df_AH171823[str(i)+"_"+str(j)] =df1

                
# df_merged = pd.concat([df_AH171823["1_2"],
#                         df_AH171823["1_3"],
#                       df_AH171823["1_5"],
#                       df_AH171823["2_3"],
#                       df_AH171823["2_5"],
#                       df_AH171823["3_5"]
#                     ], ignore_index=True, sort=False)

# all_sal_df = pickle.load( open( "all_sal_df.p", "rb" ) )
# all_sal_df["AH171823"] =df_merged
# pickle.dump( all_sal_df, open("all_sal_df.p", "wb" ) ) 
# pickle.dump( df_merged, open("AH171823_df.p", "wb" ) )  
# # ################
# # 5
# # # AH181964
# AH181964_1 = pickle.load( open( "AH181964_1.p", "rb" ) )
# AH181964_2 = pickle.load( open( "AH181964_2.p", "rb" ) )
# AH181964_3 = pickle.load( open( "AH181964_3.p", "rb" ) )
# AH181964_4 = pickle.load( open( "AH181964_4.p", "rb" ) )
# new_svd(AH181964_1)
# new_svd(AH181964_2)
# new_svd(AH181964_3)
# new_svd(AH181964_4)
# s="AH181964"
# AH181964_dict = {}
# AH181964_dict[1] = AH181964_1
# AH181964_dict[2] = AH181964_2
# AH181964_dict[3] = AH181964_3
# AH181964_dict[4] = AH181964_4
# df_AH181964 ={}


# for i in range(1,5):
#     for j in range(2,5):
#         print("-----------------------")
#         print(i)
#         print(j)
#         if(i!=j and i<j):
#             pickfilename=s+ r"_"+ str(i)+ r"_"+ s + r"_"+ str(j)
#             print(pickfilename)
#             list_a_b =pickle.load( open( pickfilename +".p", "rb" ) ) 
#             df1 =create_df(AH181964_dict[i],AH181964_dict[j],list_a_b, s+ r"_"+ str(i),s + r"_"+ str(j))
#             df_AH181964[str(i)+"_"+str(j)] =df1
                
# df_merged = pd.concat([df_AH181964["1_2"],
#                         df_AH181964["1_3"],
#                       df_AH181964["1_4"],
#                       df_AH181964["2_3"],
#                       df_AH181964["2_4"],
#                       df_AH181964["3_4"]
#                     ], ignore_index=True, sort=False)

# all_sal_df = pickle.load( open( "all_sal_df.p", "rb" ) )
# all_sal_df["AH181964"] =df_merged
# pickle.dump( all_sal_df, open("all_sal_df.p", "wb" ) ) 
# pickle.dump( df_merged, open("AH181964_df.p", "wb" ) )  
# # ####################
# # #6
# # AH181965
# AH181965_1 = pickle.load( open( "AH181965_1.p", "rb" ) )
# AH181965_2 = pickle.load( open( "AH181965_2.p", "rb" ) )
# AH181965_3 = pickle.load( open( "AH181965_3.p", "rb" ) )
# AH181965_4 = pickle.load( open( "AH181965_4.p", "rb" ) )
# AH181965_5 = pickle.load( open( "AH181965_5.p", "rb" ) )
# new_svd(AH181965_1)
# new_svd(AH181965_2)
# new_svd(AH181965_3)
# new_svd(AH181965_4)
# new_svd(AH181965_5)
# s="AH181965"
# df_AH181965 = {}
# AH181965_dict ={}

# AH181965_dict[1] = AH181965_1
# AH181965_dict[2] = AH181965_2
# AH181965_dict[3] = AH181965_3
# AH181965_dict[4] = AH181965_4
# AH181965_dict[5] = AH181965_5

# for i in range(1,6):
#     for j in range(2,6):
#         print("-----------------------")
#         print(i)
#         print(j)
#         if(i!=j and i<j):
#             pickfilename=s+ r"_"+ str(i)+ r"_"+ s + r"_"+ str(j)
#             print(pickfilename)
#             list_a_b =pickle.load( open( pickfilename +".p", "rb" ) ) 
#             df1 =create_df(AH181965_dict[i],AH181965_dict[j],list_a_b, s+ r"_"+ str(i),s + r"_"+ str(j))
#             df_AH181965[str(i)+"_"+str(j)] =df1
                
# df_merged = pd.concat([df_AH181965["1_2"],
#                         df_AH181965["1_3"],
#                       df_AH181965["1_4"],
#                       df_AH181965["1_5"],                
#                       df_AH181965["2_3"],
#                       df_AH181965["2_4"],
#                       df_AH181965["2_5"],
#                       df_AH181965["3_4"],
#                       df_AH181965["3_5"],                     
#                       df_AH181965["4_5"]
#                         ], ignore_index=True, sort=False)

# all_sal_df = pickle.load( open( "all_sal_df.p", "rb" ) )
# all_sal_df["AH181965"] =df_merged
# pickle.dump( all_sal_df, open("all_sal_df.p", "wb" ) ) 
# pickle.dump( df_merged, open("AH181965_df.p", "wb" ) )  




















#######
# AH181956
#7
# AH181956_4 = pickle.load( open( "AH181956_4.p", "rb" ) )
# AH181956_5 = pickle.load( open( "AH181956_5.p", "rb" ) )
# AH181956_6 = pickle.load( open( "AH181956_6.p", "rb" ) )

# s="AH181956"
# df_AH181956 = {}
# AH181956_dict ={}


# AH181956_dict[6] = AH181956_6
# AH181956_dict[4] = AH181956_4
# AH181956_dict[5] = AH181956_5

# for i in range(4,7):
#     for j in range(5,7):
#         print("-----------------------")
#         print(i)
#         print(j)
#         if(i!=j and i<j):
#             pickfilename=s+ r"_"+ str(i)+ r"_"+ s + r"_"+ str(j)
#             print(pickfilename)
#             list_a_b =pickle.load( open( pickfilename +".p", "rb" ) ) 
#             df1 =create_df(AH181956_dict[i],AH181956_dict[j],list_a_b)
#             df_AH181956[str(i)+"_"+str(j)] =df1
                
# df_merged = pd.concat([                     
#                       df_AH181956["4_5"],
#                       df_AH181956["5_6"],
#                       df_AH181956["4_6"],
#                         ], ignore_index=True, sort=False)

# all_sal_df = pickle.load( open( "all_sal_df.p", "rb" ) )
# all_sal_df["AH181956"] =df_merged
# pickle.dump( all_sal_df, open("all_sal_df.p", "wb" ) ) 
# pickle.dump( df_merged, open("AH181956_df.p", "wb" ) )  

###########
#8
# AH171801
# AH171801_1 = pickle.load( open( "AH171801_1.p", "rb" ) )
# AH171801_2 = pickle.load( open( "AH171801_2.p", "rb" ) )
# # AH171801_4 = pickle.load( open( "AH171801_4.p", "rb" ) )
# AH171801_5 = pickle.load( open( "AH171801_3.p", "rb" ) )
# s="AH171801"

# #neares spots
# nearest_spot(AH171801_1)
# nearest_spot(AH171801_2)
# nearest_spot(AH171801_5)
# # list_a_b_1 =pickle.load( open( "AH171801_1_AH171801_2.p", "rb" ) ) 
# # df_nearest_spots = pd.DataFrame()
# # numOfMistakes = 0;
# # for i in range(len(list_a_b_1)):
# #     spot1= list_a_b_1[i][0] 
# #     spot2= list_a_b_1[i][1]
# #     nearest_spot1 = all_near_spots[0][spot1][0]
# #     second_nearest_spot1 = all_near_spots[0][spot1][1]
# #     nearest_spot2 = all_near_spots[1][spot2][0]
# #     second_nearest_spot2 = all_near_spots[1][spot2][1]

# #     flag = ([nearest_spot1,nearest_spot2] in list_a_b_1 )or ([nearest_spot1,second_nearest_spot2] in list_a_b_1) or ([second_nearest_spot1,nearest_spot2] in list_a_b_1)
# #     # if (!flag): numOfMistakes = numOfMistakes+1
    
# #     # if(s1,s2 in nearest_spot[0])
    
# #     df_nearest_spots = df_nearest_spots.append({"spot1":spot1,"spot2":spot2,"correct":flag} , ignore_index = True) 
# # #end of nearest spots


# AH171801_dict = {}
# AH171801_dict[1] = AH171801_1
# AH171801_dict[2] = AH171801_2
# # AH171801_dict[4] = AH171801_4
# AH171801_dict[5] = AH171801_5



# df_nearest_spots = pd.DataFrame()
# AH171801 ={}
# df_AH171801 ={}
# list_1 =[1,2]
# list_2 = [2]
# num_1=-1
# num_2=0
# for i in list_1:
#     num_1=num_1+1
#     num_2=num_1
#     for j in list_2:    
#         print("-----------------------")
#         print(i)
#         print(j)
#         # if(i!=j and i<j):
#         if(i<j):
#             num_2 = num_2+1
#             pickfilename=s+ r"_"+ str(i)+ r"_"+ s + r"_"+ str(j)
#             print(pickfilename)
#             list_a_b =pickle.load( open( pickfilename +".p", "rb" ) )
#             #nearest spots
          
#             for k in range(len(list_a_b)):
#                 spot1= list_a_b[k][0] 
#                 spot2= list_a_b[k][1]
#                 try:
#                     nearest_spot1 = all_near_spots[num_1][spot1][0]
#                     second_nearest_spot1 = all_near_spots[num_1][spot1][1]
#                     nearest_spot2 = all_near_spots[num_2][spot2][0]
#                     second_nearest_spot2 = all_near_spots[num_2][spot2][1]

#                     flag = ([nearest_spot1,nearest_spot2] in list_a_b )or ([nearest_spot1,second_nearest_spot2] in list_a_b) or ([second_nearest_spot1,nearest_spot2] in list_a_b) or ([second_nearest_spot1,second_nearest_spot2] in list_a_b)
#                 # if (!flag): numOfMistakes = numOfMistakes+1
                    
#                 # if(s1,s2 in nearest_spot[0])
                
#                     df_nearest_spots = df_nearest_spots.append({"salamander1":s+ r"_"+ str(i),"spot1":spot1,"salamander2":s+ r"_"+ str(j),"spot2":spot2,"correct":flag} , ignore_index = True)
#                 except: 
#                     continue
            
            
            

#             df1 =create_df(AH171801_dict[i],AH171801_dict[j],list_a_b,  s+ r"_"+ str(i),s + r"_"+ str(j))
#             df_AH171801[str(i)+"_"+str(j)] =df1
 
# df_merged = pd.concat([df_AH171801["1_2"],
                      
#                       # df_AH171801["1_5"],
                   
#                       # df_AH171801["2_5"],
                    
#                     ], ignore_index=True, sort=False)
# # pickle.dump( df_merged, open("df_AH171801.p", "wb" ) ) 
# # all_sal_df = pickle.load( open( "all_sal_df.p", "rb" ) )
# # all_sal_df["AH171801"] =df_merged
# # pickle.dump( all_sal_df, open("all_sal_df.p", "wb" ) ) 
# pickle.dump( df_AH171801, open("AH171801_df.p", "wb" ) )  
# counter = 0
# for i in range( len(df_nearest_spots)):
#     if df_nearest_spots.correct[i]==1:
#         counter=counter+1
# percentage = counter/len(df_nearest_spots)
# print(percentage)













######################################


AH171823_1 = pickle.load( open( "AH181967_1.p", "rb" ) )
AH171823_2 = pickle.load( open( "AH181967_2.p", "rb" ) )
# AH171801_4 = pickle.load( open( "AH171801_4.p", "rb" ) )
# AH171801_5 = pickle.load( open( "AH171801_3.p", "rb" ) )
s="AH181967"

#neares spots
nearest_spot(AH171823_1)
nearest_spot(AH171823_2)
# nearest_spot(AH171801_5)
# list_a_b_1 =pickle.load( open( "AH171801_1_AH171801_2.p", "rb" ) ) 
# df_nearest_spots = pd.DataFrame()
# numOfMistakes = 0;
# for i in range(len(list_a_b_1)):
#     spot1= list_a_b_1[i][0] 
#     spot2= list_a_b_1[i][1]
#     nearest_spot1 = all_near_spots[0][spot1][0]
#     second_nearest_spot1 = all_near_spots[0][spot1][1]
#     nearest_spot2 = all_near_spots[1][spot2][0]
#     second_nearest_spot2 = all_near_spots[1][spot2][1]

#     flag = ([nearest_spot1,nearest_spot2] in list_a_b_1 )or ([nearest_spot1,second_nearest_spot2] in list_a_b_1) or ([second_nearest_spot1,nearest_spot2] in list_a_b_1)
#     # if (!flag): numOfMistakes = numOfMistakes+1
    
#     # if(s1,s2 in nearest_spot[0])
    
#     df_nearest_spots = df_nearest_spots.append({"spot1":spot1,"spot2":spot2,"correct":flag} , ignore_index = True) 
# #end of nearest spots


AH171823_dict = {}
AH171823_dict[1] = AH171823_1
AH171823_dict[2] = AH171823_2
# AH171801_dict[4] = AH171801_4
# AH171823_dict[5] = AH171801_5



df_nearest_spots = pd.DataFrame()
AH171823 ={}
df_AH171823 ={}
list_1 =[1,2]
list_2 = [2]
num_1=-1
num_2=0
for i in list_1:
    num_1=num_1+1
    num_2=num_1
    for j in list_2:    
        print("-----------------------")
        print(i)
        print(j)
        # if(i!=j and i<j):
        if(i<j):
            num_2 = num_2+1
            pickfilename=s+ r"_"+ str(i)+ r"_"+ s + r"_"+ str(j)
            print(pickfilename)
            list_a_b =pickle.load( open( pickfilename +".p", "rb" ) )
            #nearest spots
          
            for k in range(len(list_a_b)):
                spot1= list_a_b[k][0] 
                spot2= list_a_b[k][1]
                try:
                    nearest_spot1 = all_near_spots[num_1][spot1][0]
                    second_nearest_spot1 = all_near_spots[num_1][spot1][1]
                    nearest_spot2 = all_near_spots[num_2][spot2][0]
                    second_nearest_spot2 = all_near_spots[num_2][spot2][1]

                    flag = ([nearest_spot1,nearest_spot2] in list_a_b )or ([nearest_spot1,second_nearest_spot2] in list_a_b) or ([second_nearest_spot1,nearest_spot2] in list_a_b) or ([second_nearest_spot1,second_nearest_spot2] in list_a_b)
                # if (!flag): numOfMistakes = numOfMistakes+1
                    
                # if(s1,s2 in nearest_spot[0])
                
                    df_nearest_spots = df_nearest_spots.append({"salamander1":s+ r"_"+ str(i),"spot1":spot1,"salamander2":s+ r"_"+ str(j),"spot2":spot2,"correct":flag} , ignore_index = True)
                except: 
                    continue
            
            
            

            df1 =create_df(AH171823_dict[i],AH171823_dict[j],list_a_b,  s+ r"_"+ str(i),s + r"_"+ str(j))
            df_AH171823[str(i)+"_"+str(j)] =df1
 
df_merged = pd.concat([df_AH171823["1_2"],
                      
                      # df_AH171801["1_5"],
                   
                      # df_AH171801["2_5"],
                    
                    ], ignore_index=True, sort=False)
# pickle.dump( df_merged, open("df_AH171801.p", "wb" ) ) 
# all_sal_df = pickle.load( open( "all_sal_df.p", "rb" ) )
# all_sal_df["AH171801"] =df_merged
# pickle.dump( all_sal_df, open("all_sal_df.p", "wb" ) ) 
# pickle.dump( df_AH171823, open("AH171823_df.p", "wb" ) )  
counter = 0
for i in range( len(df_nearest_spots)):
    if df_nearest_spots.correct[i]==1:
        counter=counter+1
percentage = counter/len(df_nearest_spots)
print(percentage)


