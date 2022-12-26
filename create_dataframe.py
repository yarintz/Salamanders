# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 17:29:38 2022

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



list_names =[       #Salamnder A
                    #spot 1
                    "sal1","num1","pixels_num1", "norm_x1","norm_y1","rectangle_area1",           
                    "percentange_rec1", "sigma1_1","sigma2_1", "sigma_ratio1",
                    #-----------------------------------------------------------------------------------------------------
                    #spot 1 nearest negihbor
                    #1_1
                    'num1_1',
                    "pixels_num1_1", "norm_x1_1","norm_y1_1","rectangle_area1_1",           
                    "percentange_rec1_1", "sigma1_1_1","sigma2_1_1", "sigma_ratio1_1",
                    #--------------------------------------------------------------------------------------------------
                    #spot 1  second nearest negihbor
                    #1_2
                    'num1_2',
                    "pixels_num1_2", "norm_x1_2","norm_y1_2","rectangle_area1_2",           
                    "percentange_rec1_2", "sigma1_1_2","sigma2_1_2", "sigma_ratio1_2",
                    #--------------------------------------------------------------------------------------------------
                   ###############################################################################################################
                    "scale",# salamander A and salamander B
                    ##############################################################################################################
                    #Salamnder B
                    #spot 2
                    "sal2",
                    "num2","pixels_num2", "norm_x2","norm_y2","rectangle_area2",           
                    "percentange_rec2", "sigma1_2","sigma2_2", "sigma_ratio2",
                    #-------------------------------------------------------------------------------------------------------------------------
                    ##spot 2 nearest negihbor
                    # #2_1
                    'num2_1',
                    "pixels_num2_1", "norm_x2_1","norm_y2_1","rectangle_area2_1",           
                    "percentange_rec2_1", "sigma1_2_1","sigma2_2_1", "sigma_ratio2_1",
                    #-------------------------------------------------------------------------------------------------------------------------
                    ##spot 2 second nearest negihbor
                    # #2_2
                    'num2_2',
                    "pixels_num2_2", "norm_x2_2","norm_y2_2","rectangle_area2_2",           
                    "percentange_rec2_2", "sigma1_2_2","sigma2_2_2", "sigma_ratio2_2",                  
                    #--------------------------------------------------------------------------------------------------------------------------
                    #3 
                    #vectors division spot 1 / spot 2
                    "pixels_num3", "norm_x3","norm_y3","rectangle_area3",           
                    "percentange_rec3", "sigma1_3","sigma2_3", "sigma_ratio3",
                    #-------------------------------------------------------------------------------------------------------------------------
                    #4
                    # vectors subtraction
                    "pixels_num4", "norm_x4","norm_y4","rectangle_area4",           
                    "percentange_rec4", "sigma1_4","sigma2_4", "sigma_ratio4",
                    #-------------------------------------------------------------------------------------------------------------------------
                    #5 
                    #1->1'
                    #nearest neighbors relations- 
                    #vectors division
                    "pixels_num5", "norm_x5","norm_y5","rectangle_area5",           
                    "percentange_rec5", "sigma1_5","sigma2_5", "sigma_ratio5",
                    #-------------------------------------------------------------------
                    #6
                    #1->2'
                    #nearest neighbor with second nearest neighbor relations- 
                    #vectors division
                     "pixels_num6", "norm_x6","norm_y6","rectangle_area6",           
                    "percentange_rec6", "sigma1_6","sigma2_6", "sigma_ratio6",
                    #------------------------------------------------------------------
                    #7
                    # 2->1'
                    #second nearest neighbor with nearest neighbor relations- 
                    #vectors division
                    "pixels_num7", "norm_x7","norm_y7","rectangle_area7",           
                    "percentange_rec7", "sigma1_7","sigma2_7", "sigma_ratio7",
                    #--------------------------------------------------------------------------
                    ##################################################################################
                    # vectors subtraction
                    ##################################################################################
                    #8
                    # 1 -> 1
                    #nearest neighbors relations- 
                    #vectors subtraction
                    "pixels_num8", "norm_x8","norm_y8","rectangle_area8",           
                    "percentange_rec8", "sigma1_8","sigma2_8", "sigma_ratio8",
        
                    #-----------------------------------------------------------------------------------
                    #9
                    #1-> 2'
                    #nearest neighbor with second nearest neighbor relations- 
                    #vectors subtraction
                    "pixels_num9", "norm_x9","norm_y9","rectangle_area9",           
                    "percentange_rec9", "sigma1_9","sigma2_9", "sigma_ratio9",
             
                    #-------------------------------------------------------------------------------------
                    #10
                    # 2-> 1'
                    #second nearest neighbor with nearest neighbor relations- 
                    #vectors subtraction
                    "pixels_num10", "norm_x10","norm_y10","rectangle_area10",           
                    "percentange_rec10", "sigma1_10","sigma2_10", "sigma_ratio10",
                    #---------------------------------------------------------------------------------------
                    #Y#
                    "y"
                    # "num","pixels_num", "norm_x","norm_y","rectangle_area",           
                    # "percentange_rec", "sigma1","sigma2", "sigma_ratio"
                    ]

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
       
        if(len(label_1)>max_pixels):
            max_pixels = len(label_1)
            biggest_spot = label_1
    
    # print(len(biggest_spot))
            
    return len(biggest_spot)


    # sopt_point ={} # spots num value- x,y 
    # points ={}
    # all_points =[]
    # distance_list =[]
    # near_spots ={}
    # for i in range(len(s.spots.spots)):
    #     sopt_point[s.spots.spots[i].num] =[s.spots.spots[i].center_x,s.spots.spots[i].center_y]
    #     points[s.spots.spots[i].center_x,s.spots.spots[i].center_y] =s.spots.spots[i].num
    #     all_points.append([s.spots.spots[i].center_x,s.spots.spots[i].center_y])
    # for i in range(len(all_points)):
    #     p2_dis ={}
    #     dis_p2 ={}
    #     distance_list=[]
    #     for j in range(len(all_points)):
    #         if(i!=j):
    #             p1 =all_points[i]
    #             p2=all_points[j]
    #             distance = calc_dis(p1,p2)
    #             p2_dis[points[(p2[0],p2[1])]] =distance
    #             dis_p2[distance] =points[(p2[0],p2[1])]
    #             distance_list.append(distance)
    
    #     distance_list.sort() 
    #     l=[]
    #     for k in range (0,3):
    #         l.append((dis_p2[distance_list[k]]))
        
    #     near_spots[points[(p1[0],p1[1])]]   =  l#distance_list
    # return near_spots
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

    
def vactor_division (v1,v2,scale,scale_norm_x,scale_norm_y):
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
def vector_subtraction (v1,v2,scale,scale_norm_x,scale_norm_y):
    v4 = np.array([v1[0]- v2[0]*(scale**2),
                  v1[1] - np.round(v2[1]*scale_norm_x),
                   v1[2]- np.round(v2[2]*scale_norm_y),
                   v1[3]-(v2[3]*(scale**2)),
                   v1[4]-v2[4],
                   v1[5]-v2[5]*scale,
                   v1[6]-v2[6]*scale,
                   is_zero(v1[6],v1[5])- is_zero(v2[6],v2[5])
                   ])
    return v4

def crate_spots_list_train(par ,s1,s2,list_a_b):
    if(par == 0):#train
        spots_num_a = [*range(1, len(s1.spots.spots)+1, 1)]
        spots_num_b = [*range(1, len(s2.spots.spots)+1, 1)]
        
        pairs = [[w,f] for w in spots_num_a for f in spots_num_b]
        all_spots_list = list.copy(list_a_b)
        for i in range (len(list_a_b)*2):
            choice = [random.choice(pairs) for _ in range (1)]
            while([choice] in list_a_b):
                    choice = [random.choice(pairs) for _ in range (1)]  
            all_spots_list.append(choice[0])
    random.shuffle(all_spots_list)  
    return all_spots_list

def crate_spots_list_test(par ,s1,s2,list_a_b):
     spots_num_a = [*range(1, len(s1.spots.spots)+1, 1)]
     spots_num_b = [*range(1, len(s2.spots.spots)+1, 1)]     
     pairs = [[w,f] for w in spots_num_a for f in spots_num_b]
     random.shuffle(pairs)  
     return pairs
    
#     random.shuffle(pairs)
                
        #par = paramets 0-> train, 1->test
def create_df(s1,s2,list_a_b,name1,name2,par):
    

    df = pd.DataFrame(columns=list_names)
    near1 = s1.nearest_neighbors
    near2 = s2.nearest_neighbors

    # spots_num_a = [*range(1, len(s1.spots.spots)+1, 1)]
    # spots_num_b = [*range(1, len(s2.spots.spots)+1, 1)]
    
    # pairs = [[w,f] for w in spots_num_a for f in spots_num_b]
    # all_spots_list = list.copy(list_a_b)
    # for i in range (len(list_a_b)*2):
    #     choice = [random.choice(pairs) for _ in range (1)]
    #     while([choice] in list_a_b):
    #             choice = [random.choice(pairs) for _ in range (1)]  
    #     all_spots_list.append(choice[0])
        
    
    # random.shuffle(all_spots_list)  
    y = 0
    scale, choosen_sal = calc_scale(s1, s2)
    spots_list =[]   
    
    if(par == 0):
        spots_list = crate_spots_list_train(par ,s1,s2,list_a_b)
        
    if(par == 1):
        spots_list = crate_spots_list_test(par ,s1,s2,list_a_b)
        
    for i in range (len(spots_list)):

        a = spots_list[i][0]
        b = spots_list[i][1]
        ########
        #SPOT 1
        v1 =s1.spots.spots[a-1].vac
        num1 =s1.spots.spots[a-1].num
        
        # SPOT 2
        v2 =s2.spots.spots[b-1].vac
     
        num2 =s2.spots.spots[b-1].num
        ######
        near_a = near1[num1]
        #nearest neighbor 
        v1_1 =s1.spots.spots[near_a[0]-1].vac
        v1_2 =s1.spots.spots[near_a[1]-1].vac
        near_b =near2[num2]
        v2_1 =s2.spots.spots[near_b[0]-1].vac
        v2_2 =s2.spots.spots[near_b[1]-1].vac


        max_x_1= max (spot.norm_x for spot in s1.spots.spots)
        max_y_1= max (spot.norm_y for spot in s1.spots.spots)
        max_x_2= max (spot.norm_x for spot in s2.spots.spots)
        max_y_2= max (spot.norm_y for spot in s2.spots.spots)
        scale_norm_x = max_x_1/max_x_2
        scale_norm_y = max_y_1/max_y_2
        v3 = vactor_division (v1,v2,scale,scale_norm_x,scale_norm_y)
        v4 = vector_subtraction(v1,v2,scale,scale_norm_x,scale_norm_y)
        
        #nearest neighbors relations 
        #v5: 1->1'
        v5 = vactor_division (v1_1,v2_1,scale,scale_norm_x,scale_norm_y) 
        #6 1->2'
        v6 = vactor_division(v1_1,v2_2,scale,scale_norm_x,scale_norm_y) 
        #7 2->1'
        v7 = vactor_division(v1_2,v2_1,scale,scale_norm_x,scale_norm_y) 
        ####################################################################
        #8 1-1'
        v8 = vector_subtraction (v1_1,v2_1,scale,scale_norm_x,scale_norm_y) 
        #9 1->2'
        v9 = vector_subtraction (v1_1,v2_2,scale,scale_norm_x,scale_norm_y) 
        #10 2->1'
        v10 = vector_subtraction (v1_2,v2_1,scale,scale_norm_x,scale_norm_y)
            
    
        if([a,b] in list_a_b):
            y = 1
            # print(y)
        else:
            y=0
        df = df.append({
            'sal1':name1 ,'num1':a, 'pixels_num1': v1[0], 'norm_x1': v1[1],
                'norm_y1': v1[2],'rectangle_area1': v1[3],'percentange_rec1': v1[4],
                'sigma1_1':v1[5],
                'sigma2_1' : v1[6],'sigma_ratio1': is_zero(v1[6],v1[5]),
                # #1_1
                'num1_1':near_a[0], 
                'pixels_num1_1': v1_1[0], 'norm_x1_1': v1_1[1],
                'norm_y1_1': v1_1[2],'rectangle_area1_1': v1_1[3],'percentange_rec1_1': v1_1[4],
                'sigma1_1_1':v1_1[5],
                'sigma2_1_1' : v1_1[6],'sigma_ratio1_1': is_zero(v1_1[6],v1_1[5]),
                #1_2
                'num1_2':near_a[1],
                'pixels_num1_2': v1_2[0], 'norm_x1_2': v1_2[1],
                'norm_y1_2': v1_2[2],'rectangle_area1_2': v1_2[3],'percentange_rec1_2': v1_2[4],
                'sigma1_1_2':v1_2[5],
                'sigma2_1_2' : v1_2[6],'sigma_ratio1_2': is_zero(v1_2[6],v1_2[5]),
                ####################################################################
                 'scale':scale,
                #####################################################################
                 #Salamander B
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
                'num2_1':near_b[0],
                'pixels_num2_1': v2_1[0]*(scale**2), 
                'norm_x2_1': np.round(v2_1[1]*scale_norm_x),
                'norm_y2_1': np.round(v2_1[2]*scale_norm_y),
                'rectangle_area2_1': v2_1[3]*(scale**2),
                'percentange_rec2_1': v2_1[4],
                'sigma1_2_1':v2_1[5]*scale,
                'sigma2_2_1' : v2_1[6]*scale,
                'sigma_ratio2_1': is_zero(v2_1[6],v2_1[5]),
                # #2_2
                'num2_2':near_b[1],
                'pixels_num2_2': v2_2[0], 'norm_x2_2': v2_2[1],
                'norm_y2_2': v2_2[2],'rectangle_area2_2': v2_2[3],'percentange_rec2_2': v2_2[4],
                'sigma1_2_2':v2_2[5],
                'sigma2_2_2' : v2_2[6],'sigma_ratio2_2': is_zero(v2_2[6],v2_2[5]),
                ##########################################################################
                #                          VECTORS RELATIONS
                ##########################################################################
                #vectors division spot 1 / spot 2
                'pixels_num3': v3[0],
                'norm_x3': v3[1],
                'norm_y3': v3[2],
                'rectangle_area3': v3[3],
                'percentange_rec3':v3[4],
                'sigma1_3': v3[5],
                'sigma2_3': v3[6],
                'sigma_ratio3': v3[7],
                #-------------------------------------------------------------------------------
                #4 -vectors subtraction
                "pixels_num4":v4[0],
                "norm_x4":v4[1],
                "norm_y4":v4[2],
                "rectangle_area4":v4[3],           
                "percentange_rec4":v4[4],
                "sigma1_4":v4[5],
                "sigma2_4":v4[6],
                "sigma_ratio4":v4[7],
                #---------------------------------------------------------------------------------
                #nearest neighbors relations- 
                #vectors division
                #5 
                "pixels_num5":v5[0],
                "norm_x5":v5[1],
                "norm_y5":v5[2],
                "rectangle_area5":v5[3],           
                "percentange_rec5":v5[4],
                "sigma1_5":v5[5],
                "sigma2_5":v5[6],
                "sigma_ratio5":v5[7],
                #------------------------------------------------------------------------------
                #6
                #1->2'
                #nearest neighbor with second nearest neighbor relations- 
                #vectors division
                "pixels_num6":v6[0],
                "norm_x6":v6[1],
                "norm_y6":v6[2],
                "rectangle_area6":v6[3],           
                "percentange_rec6":v6[4],
                "sigma1_6":v6[5],
                "sigma2_6":v6[6],
                "sigma_ratio6":v6[7],
                #7
                # 2->1'
                #second nearest neighbor with nearest neighbor relations- 
                #vectors division
                "pixels_num7":v7[0],
                "norm_x7":v7[1],
                "norm_y7":v7[2],
                "rectangle_area7":v7[3],           
                "percentange_rec7":v7[4],
                "sigma1_7":v7[5],
                "sigma2_7":v7[6],
                "sigma_ratio7":v7[7],
                #8
                 #8
                # 1 -> 1
                #nearest neighbors relations- 
                #vectors subtraction
                "pixels_num8":v8[0],
                "norm_x8":v8[1],
                "norm_y8":v8[2],
                "rectangle_area8":v8[3],           
                "percentange_rec8":v8[4],
                "sigma1_8":v8[5],
                "sigma2_8":v8[6],
                "sigma_ratio8":v8[7],
                #-----------------------------------------------------------------------------------
                #9
                #1-> 2'
                #nearest neighbor with second nearest neighbor relations- 
                #vectors subtraction
                "pixels_num9":v9[0],
                "norm_x9":v9[1],
                "norm_y9":v9[2],
                "rectangle_area9":v9[3],           
                "percentange_rec9":v9[4],
                "sigma1_9":v9[5],
                "sigma2_9":v9[6],
                "sigma_ratio9":v9[7],
                #-------------------------------------------------------------------------------------
                #10
                # 2-> 1'
                #second nearest neighbor with nearest neighbor relations- 
                #vectors subtraction
                "pixels_num10":v10[0],
                "norm_x10":v10[1],
                "norm_y10":v10[2],
                "rectangle_area10":v10[3],           
                "percentange_rec10":v10[4],
                "sigma1_10":v10[5],
                "sigma2_10":v10[6],
                "sigma_ratio10":v10[7],
                #################################################################################################
                "y":y} , ignore_index = True) 
    # vectors_dict.clear()
    return df
#create datafreme
dataframe_all_sals ={}
names = pickle.load( open( "names.p", "rb" ) )


slamander_names = pickle.load( open( "slamander_names.p", "rb" ) )
salas_pairs=[]
sals_numbers ={}
sals_pairs_dict ={}
for sal in slamander_names.keys():
    sals_pairs_dict[sal] = []
    list_numbers = []
    for val in slamander_names[sal]:
        list_numbers.append(val[-1])
    sals_numbers[sal] = list_numbers
    # print("--------------------------------")
    # print(slamander_names[sal])
    # print(list_numbers)
    for i in range(len(list_numbers)):
        if(i+1<len(list_numbers)):
            j = i+1
            # print("j", j)
            for k in range(j ,len(list_numbers)):
                salas_pairs.append([slamander_names[sal][i],slamander_names[sal][k]])               
                if sals_pairs_dict.get(sal):
                  sals_pairs_dict[sal].append([slamander_names[sal][i],slamander_names[sal][k]])
             
                else:
                  sals_pairs_dict[sal] = [[slamander_names[sal][i],slamander_names[sal][k]]]

        
              
  
dataframe_all_sals ={}
df_name  = salas_pairs[0][1][:-2]
AH171801_df = {}
sals_df_train = {}
#train 
# for i in range(len(salas_pairs)):
#     pickfilename = str(salas_pairs[i][0])+ r"_"+ str(salas_pairs[i][1])
#     pairs = pickle.load( open( pickfilename +".p", "rb" ) ) 
#     sal1 = pickle.load( open( str(salas_pairs[i][0]) +".p", "rb" ) ) 
#     sal2 = pickle.load( open( str(salas_pairs[i][1]) +".p", "rb" ) ) 
#     df = create_df(sal1,sal2, pairs, str(salas_pairs[i][0]),str(salas_pairs[i][1]),0)

#     dataframe_all_sals[pickfilename] = df
    
#     if not salas_pairs[i][0][:-2] in sals_df_train.keys():
#         sals_df_train[salas_pairs[i][0][:-2]] = df
#     else:
#           sals_df_train[salas_pairs[i][0][:-2]] = pd.concat([ sals_df_train[salas_pairs[i][0][:-2]],
#                                                             df], ignore_index=True, sort=False)
# pickle.dump( sals_df_train, open("sals_df_train.p", "wb" ) )  
# pickle.dump( dataframe_all_sals, open("dataframe_all_sals.p", "wb" ) )
    
# sal_test ={}      
# def test_df_sal(sal):
#     list_pairs = sals_pairs_dict[sal]
#     for i in range(len(list_pairs)):
#         print(i)
#         pickfilename = str(list_pairs[i][0])+ r"_"+ str(list_pairs[i][1])
#         pairs = pickle.load( open( pickfilename +".p", "rb" ) ) 
#         sal1 = pickle.load( open( str(list_pairs[i][0]) +".p", "rb" ) ) 
#         sal2 = pickle.load( open( str(list_pairs[i][1]) +".p", "rb" ) ) 
#         df = create_df(sal1,sal2, pairs, str(list_pairs[i][0]),str(list_pairs[i][1]),1)
           
#         if not list_pairs[i][0][:-2] in sal_test.keys():
#             sal_test[list_pairs[i][0][:-2]] = df
#         else:
#               sal_test[list_pairs[i][0][:-2]] = pd.concat([ sal_test[list_pairs[i][0][:-2]],
#                                                                 df], ignore_index=True, sort=False)
            
# for i in slamander_names.keys():
#     test_df_sal(i)
        

       
test_salas = {}                         
list_test =[["AH171801","AH181964"],["AH171801","AH181967"]
            ,["AH181965","AH1819116"],["AH181967","AH181983"]
            ,["AH171823","AH1819116"],["AH171823","AH181983"]] 
dict_list_test = {}
sal_test = pickle.load( open( "sal_test.p", "rb" ) )
for i in range(len(list_test)):
    name1 = list_test[i][0]  
    name2 = list_test[i][1]  
    print(name1,name2)
    list1 = slamander_names[name1]
    list2 = slamander_names[name2]
    
    dict_list_test[i] =[list1,list2]
    
    spots_num_a = dict_list_test[i][0]
    spots_num_b = dict_list_test[i][1]
    pairs = [[w,f] for w in spots_num_a for f in spots_num_b]
    random.shuffle(pairs)
    for slals in range(len(pairs)):
        pickfilename = name1+ r"_"+ name2
        # pairs = pickle.load( open( pickfilename +".p", "rb" ) ) 
        sal1 = pickle.load( open( str(pairs[slals][0]) +".p", "rb" ) ) 
        sal2 = pickle.load( open( str(pairs[slals][1]) +".p", "rb" ) ) 
        df = create_df(sal1,sal2, [], str(pairs[slals][0]),str(pairs[slals][1]),1)
        
    
        # test_salas[name2] = df
        
        if not  pickfilename in test_salas.keys():
            test_salas[pickfilename] = df
        else:
              test_salas[pickfilename] = pd.concat([test_salas[pickfilename],
                                                                df], ignore_index=True, sort=False)
  
    
    test_salas[pickfilename] = pd.concat([test_salas[pickfilename],
                                            sal_test[str(pairs[slals][0][:-2])],
                                             sal_test[str(pairs[slals][1][:-2])]], ignore_index=True, sort=False)
    
    test_salas[pickfilename] = test_salas[pickfilename].sample(frac=1).reset_index(drop = True)
    

pickle.dump( test_salas, open("test_salas.p", "wb" ) )


    # for j in range(len(salas_pairs)):
         
        


# pickle.dump( dataframe_all_sals, open("dataframe_all_sals.p", "wb" ) )

# crate_spots_list_test(par ,s1,s2,list_a_b):