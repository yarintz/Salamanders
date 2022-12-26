# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 13:04:49 2022

@author: user
"""
import numpy as np
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
from collections import Counter
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from scipy import ndimage
from find_head import head_tail
import pickle
import cv2 
sala = pickle.load( open( "AH1819116_4.p", "rb" ) )
salb = pickle.load( open("AH1819116_5.p", "rb" ) )
# salc = pickle.load( open( "AH171801_3.p", "rb" ) )
# sald = pickle.load( open("AH171801_4.p", "rb" ) )
# sale = pickle.load( open( "AH171801_5.p", "rb" ) )
spots_a = sala.spots_img
spots_a = cv2.threshold(spots_a, 127, 255, cv2.THRESH_BINARY)[1]
spots_b = salb.spots_img
spots_b = cv2.threshold(spots_b, 127, 255, cv2.THRESH_BINARY)[1]
filename8 = r'C:\Users\yarin\Documents\salamanders\spots_numbers_new'
        
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 0
color_even = (0, 255, 0)
color = (0,0, 255)

spots_copy_c = cv2.cvtColor(sala.original_img, cv2.COLOR_GRAY2BGR)

spots_copy_d = cv2.cvtColor(salb.original_img, cv2.COLOR_GRAY2BGR)

for i in range(len(sala.spots.spots)):
    color = (0,0, 255)
    org = (sala.spots.spots[i].center_y, sala.spots.spots[i].center_x)
    num = sala.spots.spots[i].num
    if((num %2) ==0):
        color = color_even
       
    spots_copy_c = cv2.putText(spots_copy_c, str(num), org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)

filename =filename8+'\AH1819116_4.jpg'
cv2.imwrite(filename, spots_copy_c)

for i in range(len(salb.spots.spots)):
    color = (0,0, 255)
    org = (salb.spots.spots[i].center_y, salb.spots.spots[i].center_x)
    num = salb.spots.spots[i].num
    if((num %2) ==0):
        color = color_even
       
    spots_copy_d = cv2.putText(spots_copy_d, str(num), org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
    
filename8 = r'C:\Users\yarin\Documents\salamanders\spots_numbers_new'

filename = filename8+'\AH1819116_5.jpg'
cv2.imwrite(filename, spots_copy_d)

