
import numpy as np
import pandas as pd
from queue import PriorityQueue
from collections import deque
import pickle

matrix = pickle.load( open( "empty_array_test.p", "rb" ) )
# q = PriorityQueue()
q =deque() 

matrix = pd.DataFrame(matrix)
best_buddie1={}
best_buddie2={}
for i,j in  enumerate(matrix):
    if(i ==0 or j==0):
        continue
    max_value = np.max(matrix[i])

    b = np.where(matrix[i]==max_value)

    best_buddie1[i]= b[0][0]


    q.append((max_value,i))

    
columns = list(matrix)
for j in range(matrix.shape[0]): 
    if( j==0):
            continue
    max_value =0;
    for i in columns:      
    # printing the third element of the column
        if (matrix[i][j] >max_value ):
           max_value =matrix[i][j]
           index=i
    
    best_buddie2[j]= index
if (len(best_buddie2)>len(best_buddie1)):
    short_dict = best_buddie1
    long_dict = best_buddie2
else:
    short_dict = best_buddie2
    long_dict = best_buddie1 
best_buddies={'shorter':'longer',}# holds the num of pixels of the "best buddies"
best_buddiesQueue =PriorityQueue() 


for i in range (len(q)):
    buddie1 = q.popleft()[1]
    try:
        buddie2 = short_dict[buddie1]
    except:
        continue
    if (long_dict[buddie2]==buddie1):
        best_buddies[buddie1] =buddie2 
        best_buddiesQueue.put((buddie1,buddie2) )



# for i in range (max(len(best_buddie2),len(best_buddie1))):
#     try:
#         buddie = best_buddie2[i]
#     except:
#         continue
#     if (best_buddie1[buddie]==i):
#         best_buddies[i] =buddie 
# for i in range (min(len(best_buddie2),len(best_buddie1))):
#     buddie1 = q.pop()[1]
#     try:
#         buddie = best_buddie2[buddie1]
#     except:
#         continue
#     best_buudies[i] =buddie 









# sal1 = pickle.load( open( "AH171801_1.p", "rb" ) )
# sal2 = pickle.load( open( "AH171823_4.p", "rb" ) )
# data = Image.fromarray(sal1.original_img)
# data2 = Image.fromarray(sal2.original_img)


# def find_quarter(sal):
#     quarter=0
#     head = sal.head
#     print(head)
#     if(head[0]<sal.original_img.shape[0]/2 and head[1]<sal.original_img.shape[1]/2):
#         quarter = 2
#     elif(head[0]>sal.original_img.shape[0]/2 and head[1]<sal.original_img.shape[1]/2):
#         quarter = 1
#     elif(head[0]<sal.original_img.shape[0]/2 and head[1]>sal.original_img.shape[1]/2):
#         quarter = 3
#     elif(head[0]>sal.original_img.shape[0]/2 and head[1]>sal.original_img.shape[1]/2):
#         quarter = 4
    
#     return quarter


# # head = sal2.head
# # if(head[0]<sal2.original_img.shape[0]/2 and head[1]<sal2.original_img.shape[1]/2):
# #     quarter2 = 2
# # elif(head[0]>sal2.original_img.shape[0]/2 and head[1]<sal2.original_img.shape[1]/2):
# #     quarter2 = 1
# # elif(head[0]<sal2.original_img.shape[0]/2 and head[1]>sal2.original_img.shape[1]/2):
# #     quarter2 = 3
# # elif(head[0]>sal2.original_img.shape[0]/2 and head[1]>sal2.original_img.shape[1]/2):
# #     quarter2 = 4

# # print(quarter2)
# quarter1 = find_quarter(sal1)
# quarter2 = find_quarter(sal2)
# differnce = quarter2-quarter1
# # data2 = data2.rotate(90*differnce)
# #data2 = data2.rotate(-90)
# data.save('a.jpg')
# data2.save('b.jpg')

# pixels1=[]
# [pixels1.append(k) for k,v in sal1.spots.spots_group.items() if v == 10]
# pixels2=[]
# [pixels2.append(k) for k,v in sal2.spots.spots_group.items() if v == 5]

# image1 = cv2.imread('a.jpg', 3)
# for i in pixels1:
#    image1[i[0],i[1],:]= [0,0,255]
# filename = 'a.jpg'
# cv2.imwrite(filename, image1)

# image2 = cv2.imread('b.jpg', 3)
# for i in pixels2:
#    image2[i[0],i[1],:]= [255,0,0]

# filename = 'b.jpg'
# cv2.imwrite(filename, image2)
# # image2.resize(image2.shape[0], image2.shape[1],3)
# #if the image rotated by 90 degrees (the differnce is odd numbe) need to resize it
# if(not differnce % 2 ==0):
#     dim = (image2.shape[1], image2.shape[1])
#     image2 = cv2.resize(image2 ,dim, interpolation = cv2.INTER_AREA)

# data2 = Image.fromarray(image2)
# data2 = data2.rotate(90*differnce)
# data2.save('b.jpg')
# # data2 = data2.rotate(45)
# # # image1 = sal1.body_img
# # # image2 = sal2.body_img
# # image1 = Image.open(sal1.body_img)
# # image2 = Image.open(sal2.body_img)
# # image1.show()
# # image2.show()
# # # plt.imshow(image1)
# # # plt.show()
# # # plt.imshow(image2)
# # # plt.show()
# # image1 = image1.resize((426, 240))
# # image1_size = image1.size
# # image2_size = image2.size
# # new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
# # new_image.paste(image1,(0,0))
# # new_image.paste(image2,(image1_size[0],0))
# # # new_image.save("images/merged_image.jpg","JPEG")
# # # new_image.show()
# # plt.imshow(new_image)
# # plt.show()
# # images = [Image.open(x) for x in ['AH171801_2.jpeg', 'P1040947.jpg' ]]
# #images = [Image.open(x) for x in [image1, image2 ]]
# images = [Image.open(x) for x in ['a.jpg', 'b.jpg' ]]
# widths, heights = zip(*(i.size for i in images))

# total_width = sum(widths)
# max_height = max(heights)

# new_im = Image.new('RGB', (total_width, max_height))

# x_offset = 0
# for im in images:
#   new_im.paste(im, (x_offset,0))
#   x_offset += im.size[0]

# plt.imshow(new_im)
# plt.show()
# new_im.save('merged.jpg')