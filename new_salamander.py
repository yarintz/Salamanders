# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 20:10:16 2022

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


#run new salamanders and save the, as pickles.
#uncomment the relevant salamanders. It will automatically run them and save them in the right folder


# all_salamanders={}
all_salamanders_list=[] # strings of the names of the images
all_salamanders_pickels={}#key = salamander's image, value = pickle
all_salamanders_strings={}#key = salamander's name, value = all images of this salamander
# fill the right name in each variable
sal_name = "AH171801"
sal_name_path = '\\'+sal_name
path = r'C:\Users\yarin\Documents\salamanders' 
ending_body = "_background_predictions.ome.jpg"
ending_spots ="_Yellow1_predictions.ome.jpg"
all_salamanders_pickels = pickle.load( open( "all_salamanders_pickels.p", "rb" ) )


# num ="_1"
# print(path+sal_name_path+sal_name_path+num)
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal1 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal1, open(sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal1 
# all_salamanders_list.append(sal_name+num)

# num ="_2"
# print(path+sal_name+sal_name+num+sal_name+num+'.jpeg')
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal2 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal2, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal2 
# all_salamanders_list.append(sal_name+num)

# num ="_3"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal3 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal3, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal3 
# all_salamanders_list.append(sal_name+num)

# num ="_4"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal4 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal4, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal4 
# all_salamanders_list.append(sal_name+num)

# num ="_5"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal5 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal5, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal5 
# all_salamanders_list.append(sal_name+num)

# # num ="_6"
# # original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# # body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# # spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# # sal6 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# # pickle.dump( sal6, open( sal_name+num+'.p', "wb" ) )
# # all_salamanders_pickels[sal_name+num] =sal6 
# # all_salamanders_list.append(sal_name+num)


# all_salamanders_strings = pickle.load( open( "all_salamanders_strings.p", "rb" ) )

# all_salamanders_strings[sal_name] = all_salamanders_list
# pickle.dump( all_salamanders_strings, open( 'all_salamanders_strings.p', "wb" ) )
# pickle.dump( all_salamanders_pickels, open( 'all_salamanders_pickels.p', "wb" ) )
# all_salamanders_list.clear()

# sal_name = "AH1819116"
# sal_name_path = '\\'+sal_name
# path = r'C:\Users\yarin\Documents\salamanders' 
# ending_body = "_background_predictions.ome.jpg"
# ending_spots ="_Yellow1_predictions.ome.jpg"
# all_salamanders_pickels = pickle.load( open( "all_salamanders_pickels.p", "rb" ) )


# num ="_1"
# print(path+sal_name_path+sal_name_path+num)
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal1 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal1, open(sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal1 
# all_salamanders_list.append(sal_name+num)

# num ="_2"
# print(path+sal_name+sal_name+num+sal_name+num+'.jpeg')
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal2 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal2, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal2 
# all_salamanders_list.append(sal_name+num)

# num ="_3"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal3 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal3, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal3 
# all_salamanders_list.append(sal_name+num)

# num ="_4"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal4 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal4, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal4 
# all_salamanders_list.append(sal_name+num)

# num ="_5"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal5 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal5, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal5 
# all_salamanders_list.append(sal_name+num)

# # num ="_6"
# # original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# # body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# # spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# # sal6 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# # pickle.dump( sal6, open( sal_name+num+'.p', "wb" ) )
# # all_salamanders_pickels[sal_name+num] =sal6 
# # all_salamanders_list.append(sal_name+num)


# all_salamanders_strings = pickle.load( open( "all_salamanders_strings.p", "rb" ) )

# all_salamanders_strings[sal_name] = all_salamanders_list
# pickle.dump( all_salamanders_strings, open( 'all_salamanders_strings.p', "wb" ) )
# pickle.dump( all_salamanders_pickels, open( 'all_salamanders_pickels.p', "wb" ) )
# all_salamanders_list.clear()



# sal_name = "AH171823"
# sal_name_path = '\\'+sal_name
# path = r'C:\Users\yarin\Documents\salamanders' 
# ending_body = "_background_predictions.ome.jpg"
# ending_spots ="_Yellow1_predictions.ome.jpg"
# all_salamanders_pickels = pickle.load( open( "all_salamanders_pickels.p", "rb" ) )


# num ="_1"
# print(path+sal_name_path+sal_name_path+num)
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal1 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal1, open(sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal1 
# all_salamanders_list.append(sal_name+num)

# num ="_2"
# print(path+sal_name+sal_name+num+sal_name+num+'.jpeg')
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal2 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal2, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal2 
# all_salamanders_list.append(sal_name+num)

# num ="_3"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal3 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal3, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal3 
# all_salamanders_list.append(sal_name+num)

# num ="_4"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal4 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal4, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal4 
# all_salamanders_list.append(sal_name+num)

# num ="_5"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal5 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal5, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal5 
# all_salamanders_list.append(sal_name+num)

# # num ="_6"
# # original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# # body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# # spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# # sal6 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# # pickle.dump( sal6, open( sal_name+num+'.p', "wb" ) )
# # all_salamanders_pickels[sal_name+num] =sal6 
# # all_salamanders_list.append(sal_name+num)


# all_salamanders_strings = pickle.load( open( "all_salamanders_strings.p", "rb" ) )

# all_salamanders_strings[sal_name] = all_salamanders_list
# pickle.dump( all_salamanders_strings, open( 'all_salamanders_strings.p', "wb" ) )
# pickle.dump( all_salamanders_pickels, open( 'all_salamanders_pickels.p', "wb" ) )
# all_salamanders_list.clear()

# sal_name = "AH181965"
# sal_name_path = '\\'+sal_name
# path = r'C:\Users\yarin\Documents\salamanders' 
# ending_body = "_background_predictions.ome.jpg"
# ending_spots ="_Yellow1_predictions.ome.jpg"
# all_salamanders_pickels = pickle.load( open( "all_salamanders_pickels.p", "rb" ) )


# num ="_1"
# print(path+sal_name_path+sal_name_path+num)
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal1 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal1, open(sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal1 
# all_salamanders_list.append(sal_name+num)

# num ="_2"
# print(path+sal_name+sal_name+num+sal_name+num+'.jpeg')
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal2 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal2, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal2 
# all_salamanders_list.append(sal_name+num)

# num ="_3"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal3 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal3, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal3 
# all_salamanders_list.append(sal_name+num)

# num ="_4"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal4 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal4, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal4 
# all_salamanders_list.append(sal_name+num)

# num ="_5"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal5 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal5, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal5 
# all_salamanders_list.append(sal_name+num)

# # num ="_6"
# # original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# # body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# # spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# # sal6 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# # pickle.dump( sal6, open( sal_name+num+'.p', "wb" ) )
# # all_salamanders_pickels[sal_name+num] =sal6 
# # all_salamanders_list.append(sal_name+num)


# all_salamanders_strings = pickle.load( open( "all_salamanders_strings.p", "rb" ) )

# all_salamanders_strings[sal_name] = all_salamanders_list
# pickle.dump( all_salamanders_strings, open( 'all_salamanders_strings.p', "wb" ) )
# pickle.dump( all_salamanders_pickels, open( 'all_salamanders_pickels.p', "wb" ) )
# all_salamanders_list.clear()

# sal_name = "AH181964"
# sal_name_path = '\\'+sal_name
# path = r'C:\Users\yarin\Documents\salamanders' 
# ending_body = "_background_predictions.ome.jpg"
# ending_spots ="_Yellow1_predictions.ome.jpg"
# all_salamanders_pickels = pickle.load( open( "all_salamanders_pickels.p", "rb" ) )


# num ="_1"
# print(path+sal_name_path+sal_name_path+num)
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal1 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal1, open(sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal1 
# all_salamanders_list.append(sal_name+num)

# num ="_2"
# print(path+sal_name+sal_name+num+sal_name+num+'.jpeg')
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal2 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal2, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal2 
# all_salamanders_list.append(sal_name+num)

# num ="_3"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal3 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal3, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal3 
# all_salamanders_list.append(sal_name+num)

# num ="_4"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal4 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal4, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal4 
# all_salamanders_list.append(sal_name+num)

# num ="_5"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal5 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal5, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal5 
# all_salamanders_list.append(sal_name+num)

# num ="_6"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal6 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal6, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal6 
# all_salamanders_list.append(sal_name+num)


# all_salamanders_strings = pickle.load( open( "all_salamanders_strings.p", "rb" ) )

# all_salamanders_strings[sal_name] = all_salamanders_list
# pickle.dump( all_salamanders_strings, open( 'all_salamanders_strings.p', "wb" ) )
# pickle.dump( all_salamanders_pickels, open( 'all_salamanders_pickels.p', "wb" ) )
# all_salamanders_list.clear()


# sal_name = "AH181983"
# sal_name_path = '\\'+sal_name
# path = r'C:\Users\yarin\Documents\salamanders' 
# ending_body = "_background_predictions.ome.jpg"
# ending_spots ="_Yellow1_predictions.ome.jpg"
# all_salamanders_pickels = pickle.load( open( "all_salamanders_pickels.p", "rb" ) )


# num ="_1"
# print(path+sal_name_path+sal_name_path+num)
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal1 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal1, open(sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal1 
# all_salamanders_list.append(sal_name+num)

# num ="_2"
# print(path+sal_name+sal_name+num+sal_name+num+'.jpeg')
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal2 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal2, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal2 
# all_salamanders_list.append(sal_name+num)

# num ="_3"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal3 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal3, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal3 
# all_salamanders_list.append(sal_name+num)

# num ="_4"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal4 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal4, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal4 
# all_salamanders_list.append(sal_name+num)

# num ="_5"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal5 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal5, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal5 
# all_salamanders_list.append(sal_name+num)

# num ="_6"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal6 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal6, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal6 
# all_salamanders_list.append(sal_name+num)


# all_salamanders_strings = pickle.load( open( "all_salamanders_strings.p", "rb" ) )

# all_salamanders_strings[sal_name] = all_salamanders_list
# pickle.dump( all_salamanders_strings, open( 'all_salamanders_strings.p', "wb" ) )
# pickle.dump( all_salamanders_pickels, open( 'all_salamanders_pickels.p', "wb" ) )
# all_salamanders_list.clear()

# sal_name = "AH181967"
# sal_name_path = '\\'+sal_name
# path = r'C:\Users\yarin\Documents\salamanders' 
# ending_body = "_background_predictions.ome.jpg"
# ending_spots ="_Yellow1_predictions.ome.jpg"
# all_salamanders_pickels = pickle.load( open( "all_salamanders_pickels.p", "rb" ) )


# num ="_1"
# print(path+sal_name_path+sal_name_path+num)
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal1 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal1, open(sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal1 
# all_salamanders_list.append(sal_name+num)

# num ="_2"
# print(path+sal_name+sal_name+num+sal_name+num+'.jpeg')
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal2 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal2, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal2 
# all_salamanders_list.append(sal_name+num)

# num ="_3"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal3 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal3, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal3 
# all_salamanders_list.append(sal_name+num)

# num ="_4"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal4 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal4, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal4 
# all_salamanders_list.append(sal_name+num)

# num ="_5"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal5 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal5, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal5 
# all_salamanders_list.append(sal_name+num)

# num ="_6"
# original_img = cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+'.jpeg',0)
# body_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_body,0)
# spots_img =cv2.imread(path+sal_name_path+sal_name_path+num+sal_name_path+num+ending_spots,0)

# sal6 = ssc.salamander(original_img, body_img,spots_img,path+sal_name_path+sal_name_path+num,sal_name+num )
# pickle.dump( sal6, open( sal_name+num+'.p', "wb" ) )
# all_salamanders_pickels[sal_name+num] =sal6 
# all_salamanders_list.append(sal_name+num)


# all_salamanders_strings = pickle.load( open( "all_salamanders_strings.p", "rb" ) )

# all_salamanders_strings[sal_name] = all_salamanders_list
# pickle.dump( all_salamanders_strings, open( 'all_salamanders_strings.p', "wb" ) )
# pickle.dump( all_salamanders_pickels, open( 'all_salamanders_pickels.p', "wb" ) )
# all_salamanders_list.clear()




























# original_img = cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH181965\AH181965_1\AH181965_1.jpeg',0)
# body_img =cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH181965\AH181965_1\AH181965_1_background_predictions.ome.jpg',0)
# spots_img =cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH181965\AH181965_1\AH181965_1_Yellow1_predictions.ome.jpg', 0)

# sal1 = ssc.salamander(original_img, body_img,spots_img,r'C:\Users\yarin\Documents\salamanders\AH181965\AH181965_1')

# pickle.dump( sal1, open( "AH181965_1.p", "wb" ) )

# original_img = cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH181965\AH181965_2\AH181965_2.jpeg',0)
# body_img =cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH181965\AH181965_2\AH181965_2_background_predictions.ome.jpg',0)
# spots_img =cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH181965\AH181965_2\AH181965_2_Yellow1_predictions.ome.jpg', 0)

# sal2 = ssco.salamander(original_img, body_img,spots_img,r'C:\Users\yarin\Documents\salamanders\AH181965\AH181965_2')

# pickle.dump( sal2, open( "AH181965_2.p", "wb" ) )
# # #___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# # #
# original_img = cv2.imread(r'C:\Users\yarin\Documents\salamanders\BT0084\BT0084_3\BT0084_3.jpeg',0)
# body_img =cv2.imread(r'C:\Users\yarin\Documents\salamanders\BT0084\BT0084_3\BT0084_3_background_predictions.ome.jpg',0)
# spots_img =cv2.imread(r'C:\Users\yarin\Documents\salamanders\BT0084\BT0084_3\BT0084_3_Yellow1_predictions.ome.jpg',0)

# # sal3 = ssc.salamander(original_img, body_img,spots_img,r'C:\Users\yarin\Documents\salamanders\BT0084\BT0084_3')

# # pickle.dump( sal3, open( "BT0084_3.p", "wb" ) )


# original_img4 = cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH171801\AH171801_4\AH171801_4.jpg',0)
# body_img4 = cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH171801\AH171801_4\AH171801_4_background_predictions.ome.jpg',0)
# spots_img4 = cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH171801\AH171801_4\AH171801_4_Yellow1_predictions.ome.jpg',0)

# # sal4 = ssc.salamander(original_img4, body_img4,spots_img4,r'C:\Users\yarin\Documents\salamanders\AH171801\AH171801_4')

# # pickle.dump( sal4, open( "AH171801_4.p", "wb" ) )

# original_img4 = cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH171801\AH171801_5\AH171801_5.jpeg',0)
# body_img4 = cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH171801\AH171801_5\AH171801_5_background_predictions.ome.jpg',0)
# spots_img4 = cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH171801\AH171801_5\AH171801_5_Yellow1_predictions.ome.jpg',0)
# #filename = r'C:\Users\yarin\Documents\salamanders\test.jpg'
# #cv2.imwrite(filename, spots_img4)
# # sal5 = ssc.salamander(original_img4, body_img4,spots_img4,r'C:\Users\yarin\Documents\salamanders\AH171801\AH171801_5')

# # pickle.dump( sal5, open( "AH171801_5.p", "wb" ) )

# original_img4 = cv2.imread(r'C:\Users\yarin\Documents\salamanders\BT0068\BT0068_6\BT0068_6.jpg',0)
# body_img4 = cv2.imread(r'C:\Users\yarin\Documents\salamanders\BT0068\BT0068_6\BT0068_6_background_predictions.ome.jpg',0)
# spots_img4 = cv2.imread(r'C:\Users\yarin\Documents\salamanders\BT0068\BT0068_6\BT0068_6_Yellow1_predictions.ome.jpg',0)
#filename = r'C:\Users\yarin\Documents\salamanders\test.jpg'
#cv2.imwrite(filename, spots_img4)
# sal6 = ssc.salamander(original_img4, body_img4,spots_img4,r'C:\Users\yarin\Documents\salamanders\BT0068\BT0068_6')

# pickle.dump( sal6, open( "BT0068_6.p", "wb" ) )


# original_img4 = cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH171801\AH171801_1\AH171801.jpeg',0)
# body_img4 = cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH171801\AH171801_1\AH171801_1.jpg',0)
# spots_img4 = cv2.imread(r'C:\Users\yarin\Documents\salamanders\AH171801\AH171801_1\AH171801_1_only_spots.ome.jpg',0)
# #filename = r'C:\Users\yarin\Documents\salamanders\test.jpg'
# #cv2.imwrite(filename, spots_img4)
# sal7 = ssc.salamander(original_img4, body_img4,spots_img4,r'C:\Users\yarin\Documents\salamanders\AH171801\AH171801_1')

# pickle.dump( sal7, open( "AH171801_1.p", "wb" ) )
# spots_copy_c = cv2.cvtColor(spots_img3, cv2.COLOR_GRAY2BGR)

# spots_copy_d = cv2.cvtColor(spots_img4, cv2.COLOR_GRAY2BGR)


# # for i in range(spots_white.shape[0]):
# #         spots_copy_a[spots_white[i][0],spots_white[i][1],:] =255
        
# font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale = 0.5
# thickness = 0
# color_even = (0, 255, 0)
# color = (0,0, 255)


# for i in range(len(sal3.spots.spots)):
#     color = (0,0, 255)
#     org = (sal3.spots.spots[i].center_y, sal3.spots.spots[i].center_x)
#     num = sal1.spots.spots[i].num
#     if((num %2) ==0):
#        color = color_even
       
#     spots_copy_c = cv2.putText(spots_copy_c, str(num), org, font, 
#                        fontScale, color, thickness, cv2.LINE_AA)

# filename = 'spots_with_numbers3.jpg'
# cv2.imwrite(filename, spots_copy_c)

# for i in range(len(sal4.spots.spots)):
#     color = (0,0, 255)
#     org = (sal4.spots.spots[i].center_y, sal4.spots.spots[i].center_x)
#     num = sal2.spots.spots[i].num
#     if((num %2) ==0):
#        color = color_even
       
#     spots_copy_d = cv2.putText(spots_copy_d, str(num), org, font, 
#                        fontScale, color, thickness, cv2.LINE_AA)
    

# filename = 'spots_with_numbers4.jpg'
# cv2.imwrite(filename, spots_copy_d)