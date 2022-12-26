# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 17:48:14 2022

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
# with open("sal1.bin", "rb") as f: # "rb" because we want to read in binary mode
#     sal1 = pickle.load(f)

# sal1 = pickle.load( open( "sal1.bin", "rb" ) )
# with open("sal1.bin", "rb") as f:
#     sal = pickle.load(f)
   


sal1 = pickle.load( open( "sal1.p", "rb" ) )

sal2 = pickle.load( open( "sal2.p", "rb" ) )

# salb = pickle.load( open( "sal2.p", "rb" ) )

spots_a = sal1.spots.spots
spots_b = sal2.spots.spots

def calc_scale_delta(x1,y1,x2,y2):
    scale = x1/x2
    delta_y = y1 -(scale*(y2)) #abs
    return scale, delta_y



match1_2 =[[47,2],
            [48,4],
            [45,3],
            [46,7],
            [44,6],
            [41,8],
            [40,12],
            [38,10],
            [39,5],
            [42,18],
            [43,20],
            [35,15],
            [37,22],
            [31,19],
            [27,21],
            [32,11],
            [28,14],
            [20,17],
            [33,24],
            [29,28],
            [21,27],
            [19,23],
            [16,29],
            [9,25],
            [8,20],
            [5,16],
            [4,13],
            [3,9],
            [23,33],
            [25,32],
            [30,31],
            [26,30],
            [22,35],
            [24,39],
            [34,40],
            [17,37],
            [13,26],
            [11,38],
            [15,41],
            [18,43],
            [12,44],
            [14,46],
            [10,47],
            [7,48],
            [6,49],
            [2,50],
            [1,51],
            ]
#sal1_sal2 = sorted(sal1_sal2, key=lambda row: (row[0]), reverse=False)

# match1_2 = match1_2.sort()
match1_2.sort()
match1_2_array = np.array(match1_2)
scale_delta_y = np.empty((0,2), int)
scale =[]
delta_y =[]
scale_delta_y_match = np.empty((0,2), int)
scale_match =[]
delta_y_match =[]

for i in range(len(match1_2)):
   s, y = calc_scale_delta(spots_a[match1_2[i][0]-1].norm_x,spots_a[match1_2[i][0]-1].norm_y,spots_b[match1_2[i][1]-1].norm_x,spots_b[match1_2[i][1]-1].norm_y)
   scale_delta_y_match = np.append(scale_delta_y, np.array([[s,y]]), axis=0)
   scale_match.append(s)
   delta_y_match.append(y)
    
for i in range(len(spots_a)):
    for j in range(len(spots_b)):
            s, y = calc_scale_delta(spots_a[i].norm_x,spots_a[i].norm_y,spots_b[j].norm_x,spots_b[j].norm_y)
            scale_delta_y = np.append(scale_delta_y, np.array([[s,y]]), axis=0)
            scale.append(s)
            delta_y.append(y)


import numpy as np
import matplotlib.pyplot as plt


N = len(scale)
x = np.array(scale)
y = np.array(delta_y)
colors = 'hotpink'
area = np.pi*3


plt.scatter(x, y, s=area, c=colors, alpha=0.5)

# Create data
N = len(scale_match)
x= np.array(scale_match)
y = np.array(delta_y_match)
color = '#88c999'
area = np.pi*3

# Plot
plt.scatter(x, y, s=area, c=color, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()






# print(sal1.head)




# print(sal2.head)