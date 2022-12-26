import pickle
import cv2 
import matplotlib.pyplot as plt
from PIL import Image
import sys
import imutils
from random import randint
import numpy as np


sal1 = pickle.load( open( "AH181983_3.p", "rb" ) )
sal1_spots = sal1.spots.num_spots
sal2 = pickle.load( open( "AH181983_4.p", "rb" ) )
sal2_spots = sal2.spots.num_spots
best_buddies = pickle.load( open( "all_best_buddies.p", "rb" ) )
# data = Image.fromarray(sal1.original_img)
# data2 = Image.fromarray(sal2.original_img)


def find_quarter(sal):
    quarter=0
    head = sal.head
    print(head)
    if(head[0]<sal.original_img.shape[0]/2 and head[1]<sal.original_img.shape[1]/2):
        quarter = 2
    elif(head[0]>sal.original_img.shape[0]/2 and head[1]<sal.original_img.shape[1]/2):
        quarter = 1
    elif(head[0]<sal.original_img.shape[0]/2 and head[1]>sal.original_img.shape[1]/2):
        quarter = 3
    elif(head[0]>sal.original_img.shape[0]/2 and head[1]>sal.original_img.shape[1]/2):
        quarter = 4
    
    return quarter

# colors = []
colors=np.round(np.random.rand(20,3)*255).astype(np.int64) #randomized color list of 20 colors
# for i in range(10):
#     colors.append('#%06X' % randint(0, 0xFFFFFF))
# colors = [[0,255,0],[255,0,0],[0,0,255],[0,150,150],[150,255,0]]
# for i in range(5):
# colors.append([0,255,0])
# head = sal2.head
# if(head[0]<sal2.original_img.shape[0]/2 and head[1]<sal2.original_img.shape[1]/2):
#     quarter2 = 2
# elif(head[0]>sal2.original_img.shape[0]/2 and head[1]<sal2.original_img.shape[1]/2):
#     quarter2 = 1
# elif(head[0]<sal2.original_img.shape[0]/2 and head[1]>sal2.original_img.shape[1]/2):
#     quarter2 = 3
# elif(head[0]>sal2.original_img.shape[0]/2 and head[1]>sal2.original_img.shape[1]/2):
#     quarter2 = 4

# print(quarter2)
# quarter1 = find_quarter(sal1)
# quarter2 = find_quarter(sal2)
# differnce = quarter2-quarter1

# data.save('a.jpg')
# data2.save('b.jpg')

for n in range(0,12):
    # sal1 = pickle.load( open( "AH171801_1.p", "rb" ) )
    # sal1_spots = sal1.spots.num_spots
    # sal2 = pickle.load( open( "AH171801_2.p", "rb" ) )
    # sal2_spots = sal2.spots.num_spots
    # best_buddies = pickle.load( open( "all_best_buddies.p", "rb" ) )
    
    data = Image.fromarray(sal1.original_img)
    data2 = Image.fromarray(sal2.original_img)
    quarter1 = find_quarter(sal1)
    quarter2 = find_quarter(sal2)
    differnce = quarter2-quarter1
    
    data.save('a.jpg')
    data2.save('b.jpg')
    num=0
    for key in best_buddies["AH181983_3","AH181983_4",n]:
        # print(key)
    
        # print(best_buddies["AH171801_1","AH171801_2",0][key])
        if(sal1_spots<sal2_spots):
            short_sal= sal1
            long_sal=sal2
        else:
            short_sal= sal2
            long_sal=sal1 
        pixels_short=[]
        [pixels_short.append(k) for k,v in short_sal.spots.spots_group.items() if v == key]
        pixels_long=[]
        [pixels_long.append(k) for k,v in long_sal.spots.spots_group.items() if v == best_buddies["AH181983_3","AH181983_4",n][key]]
        num+=1
        if(num==12):
            break
        image2 = cv2.imread('b.jpg', 3)
        image1 = cv2.imread('a.jpg', 3)
        if(sal1_spots<sal2_spots):
            pixels_1 = pixels_short 
            pixels_2 = pixels_long
        else:
            pixels_1 = pixels_long 
            pixels_2 = pixels_short
        for i in pixels_1:
           # image1[i[0],i[1],:]= [255,0,0]
           image1[i[0],i[1],]= colors[num]
        filename = 'a.jpg'
        cv2.imwrite(filename, image1)
        
    # colors[num]
        for i in pixels_2:
           image2[i[0],i[1],]= [colors[num][2],colors[num][1],colors[num][0]]
        filename = 'b.jpg'
        cv2.imwrite(filename, image2)
    # image2.resize(image2.shape[0], image2.shape[1],3)
    #if the image rotated by 90 degrees (the differnce is odd numbe) need to resize it
    if(not differnce % 2 ==0):
        dim = (image2.shape[1], image2.shape[1])
        image2 = cv2.resize(image2 ,dim, interpolation = cv2.INTER_AREA)
    
    data2 = Image.fromarray(image2)
    data2 = data2.rotate(90*differnce)
    data2.save('b.jpg')
    # data2 = data2.rotate(45)
    # # image1 = sal1.body_img
    # # image2 = sal2.body_img
    # image1 = Image.open(sal1.body_img)
    # image2 = Image.open(sal2.body_img)
    # image1.show()
    # image2.show()
    # # plt.imshow(image1)
    # # plt.show()
    # # plt.imshow(image2)
    # # plt.show()
    # image1 = image1.resize((426, 240))
    # image1_size = image1.size
    # image2_size = image2.size
    # new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
    # new_image.paste(image1,(0,0))
    # new_image.paste(image2,(image1_size[0],0))
    # # new_image.save("images/merged_image.jpg","JPEG")
    # # new_image.show()
    # plt.imshow(new_image)
    # plt.show()
    # images = [Image.open(x) for x in ['AH171801_2.jpeg', 'P1040947.jpg' ]]
    #images = [Image.open(x) for x in [image1, image2 ]]
    images = [Image.open(x) for x in ['a.jpg', 'b.jpg' ]]
    widths, heights = zip(*(i.size for i in images))
    
    total_width = sum(widths)
    max_height = max(heights)
    
    new_im = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    
    plt.imshow(new_im)
    plt.show()
    new_im.save('merged_'+str(n+1)+'.jpg')