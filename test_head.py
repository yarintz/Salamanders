# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
import cv2 
# import salamander_spots_class
# df = pickle.load( open( "AH171801_df.p", "rb" ) )
sala = pickle.load( open("AH181965_1.p", "rb" ) )
print(sala.head)
print(sala.tail)

# print(salb.head)
# print(salb.tail)
#same_spots = sala["1_2"][sala["1_2"].y==1]

# sala = pickle.load( open( "AH171801_1.p", "rb" ) )
# salb = pickle.load( open("AH171801_2.p", "rb" ) )\
# same_spots = df["1_2"][df["1_2"].y==1]