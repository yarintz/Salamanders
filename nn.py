# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 13:53:46 2022

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
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib
import numpy as np
import matplotlib.pyplot as pltts
import collections
from collections import deque
from queue import PriorityQueue
#columns 
all_best_buddies_queue = {}
all_best_buddies = {}
salamander1 = ["sal1","num1","pixels_num1", "norm_x1","norm_y1","rectangle_area1",           
                    "percentange_rec1", "sigma1_1","sigma2_1", "sigma_ratio1"]

salamander2 =[ "scale", "sal2","num2","pixels_num2", "norm_x2","norm_y2","rectangle_area2",           
                    "percentange_rec2", "sigma1_2","sigma2_2", "sigma_ratio2"]

division =["pixels_num3", "norm_x3","norm_y3","rectangle_area3",           
                    "percentange_rec3", "sigma1_3","sigma2_3", "sigma_ratio3"]

subtraction =["pixels_num4", "norm_x4","norm_y4","rectangle_area4",           
                    "percentange_rec4", "sigma1_4","sigma2_4", "sigma_ratio4"]
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

sal1_nn = [#'num1_1',
           "pixels_num1_1", "norm_x1_1","norm_y1_1","rectangle_area1_1",           
                    "percentange_rec1_1", "sigma1_1_1","sigma2_1_1", "sigma_ratio1_1"]

sal1_2nn =[#"num1_2",
           "pixels_num1_2", "norm_x1_2","norm_y1_2","rectangle_area1_2",           
                    "percentange_rec1_2", "sigma1_1_2","sigma2_1_2", "sigma_ratio1_2"]

sal2_nn =[#'num2_1',
          "pixels_num2_1", "norm_x2_1","norm_y2_1","rectangle_area2_1",           
                    "percentange_rec2_1", "sigma1_2_1","sigma2_2_1", "sigma_ratio2_1"]

sal2_2nn=[#'num2_2',
          "pixels_num2_2", "norm_x2_2","norm_y2_2","rectangle_area2_2",           
                    "percentange_rec2_2", "sigma1_2_2","sigma2_2_2", "sigma_ratio2_2"]
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#1->1'
nn_division =["pixels_num5", "norm_x5","norm_y5","rectangle_area5",           
                    "percentange_rec5", "sigma1_5","sigma2_5", "sigma_ratio5"]

nn_subtraction =["pixels_num8", "norm_x8","norm_y8","rectangle_area8",           
                    "percentange_rec8", "sigma1_8","sigma2_8", "sigma_ratio8"]
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#1->2'
# nearest neighbor with second nearest neighbor relations- 
# vectors division
nn_division_2n =[ "pixels_num6", "norm_x6","norm_y6","rectangle_area6",           
                    "percentange_rec6", "sigma1_6","sigma2_6", "sigma_ratio6"]

#vectors subtraction
nn_subtraction_2n =["pixels_num9", "norm_x9","norm_y9","rectangle_area9",           
                    "percentange_rec9", "sigma1_9","sigma2_9", "sigma_ratio9"]
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#2->1'
#second nearest neighbor with nearest neighbor relations- 
#vectors division
second_division_1n =["pixels_num7", "norm_x7","norm_y7","rectangle_area7",           
                    "percentange_rec7", "sigma1_7","sigma2_7", "sigma_ratio7"]
#vectors subtraction
second_subtraction_1n= ["pixels_num10", "norm_x10","norm_y10","rectangle_area10",           
                    "percentange_rec10", "sigma1_10","sigma2_10", "sigma_ratio10",]
y =["y"]
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#only the salamanders spots 
spots = salamander1 + salamander2
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#division
division_df = spots + division + y
#subtraction
subtraction_df = spots + subtraction + y
#division and subtraction
division_subtraction_df =spots + division+ subtraction + y
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#nearest neighbors
division_nn_df = salamander1 +sal1_nn+ salamander2 + sal2_nn + division +nn_division +y
subtraction_nn_df = salamander1 +sal1_nn+ salamander2 + sal2_nn + subtraction +nn_subtraction +y
division_subtraction_nn_df = salamander1 +sal1_nn+ salamander2 + sal2_nn + division+ subtraction +nn_division + nn_subtraction + y
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#1-2'
division_12_df = salamander1 +sal1_nn+ salamander2 + sal2_2nn + division +nn_division_2n +y
subtraction_12_df =salamander1 +sal1_nn+ salamander2 + sal2_2nn + subtraction +nn_subtraction_2n + y
division_subtraction_12_df = salamander1 +sal1_nn+ salamander2 + sal2_2nn + division+ subtraction + nn_division_2n + nn_subtraction_2n +y
#2->1'
division_21_df = salamander1 +sal1_2nn+ salamander2 + sal2_nn + division +second_division_1n +y
subtraction_21_df = salamander1 +sal1_2nn+ salamander2 + sal2_nn + subtraction +second_subtraction_1n +y
division_subtraction_21_df =salamander1 +sal1_2nn+ salamander2 + sal2_nn + division + subtraction  +second_division_1n +second_subtraction_1n+y

parameters = [division_df,subtraction_df,division_subtraction_df,
              division_nn_df,subtraction_nn_df,division_subtraction_nn_df,
              division_12_df,subtraction_12_df,division_subtraction_12_df,
              division_21_df,subtraction_21_df,division_subtraction_21_df]

parameters_names = ["division_df","subtraction_df","division_subtraction_df",
              "division_nn_df","subtraction_nn_df","division_subtraction_nn_df",
              "division_12_df","subtraction_12_df","division_subtraction_12_df",
              "division_21_df","subtraction_21_df","division_subtraction_21_df"]

###################################################################################################################################################################################################################################################

#test 
all_sal_df_test = pickle.load( open( "test_salas.p", "rb" ) )

list_test =[["AH171801","AH181964"],["AH171801","AH181967"]
            ,["AH181965","AH1819116"],["AH181967","AH181983"]
            ,["AH171823","AH1819116"],["AH171823","AH181983"]] 

# #train 
all_train = pickle.load( open( "sals_df_train.p", "rb" ) )
# train = pd.concat([ all_tain["AH181967"], all_tain["AH181965"],all_tain["AH1819116"],
#                    all_tain["AH171823"],all_tain["AH181983"]], ignore_index=True, sort=False)


def create_train_df (sal1, sal2):
    list_sals_train =[]
    for i in all_train.keys():
        if i!= sal1 and i!= sal2:
            list_sals_train.append(i)
    train = pd.concat([ all_train[list_sals_train[0]], all_train[list_sals_train[1]],
                       all_train[list_sals_train[2]],
                  all_train[list_sals_train[3]],all_train[list_sals_train[4]]], ignore_index=True, sort=False)
    return train

def remove_columns (df, par):
    new_df = df[par]
    return new_df
    
def create_test_df(sal1, sal2):
    test =all_sal_df_test[sal1+"_"+sal2]
    return test 
#nn model 
#returns the score of the modle 
#result a list of the result of the classifier - the y it predicted (1 for a match 0 for not a match)
# per - Probability estimates.-> the probabilty of every 2 spots to be in the same (1) or not (0)
#Note the per is 2d array so per[:,0] -> the probability to be classifies as 0. and per[:,1] the probabilty to be clasified as 1
#link with more info 
#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
def nn_model(train_x, train_y, test_x, test_y):
    clf = MLPClassifier(random_state=1,  alpha=0.01, hidden_layer_sizes=(60,30,10)
                    ).fit(train_x, train_y)
    clf.predict_proba(test_x)
    clf.predict(test_x)
    per = clf.predict_proba(test_x)
    results = clf.predict(test_x)
    score = clf.score(test_x, test_y)
    return per, results ,score

def count_one_colimns(empty_array):
    #num columns contains 1
    counter_ones_columns = 0
    columns_ones = 0
    list_columns = []
    counter=0
    for c in range(empty_array.shape[1]):
        col = empty_array[:, c]
        if(1 in col):
            columns_ones+=1
            # print(r)
            list_columns.append(c)
            counter = collections.Counter(col)
            counter_ones_columns += counter[1]
            
    return columns_ones, counter_ones_columns

def count_one_rows(empty_array):
#num rows contains 1
    rows_ones_counter = 0
    rows_ones = 0
    list_rows = []
    counter=0
    for r in empty_array:
        # row = empty_array[:, i]
        if(1 in r):
            rows_ones+=1
            # print(t)
            # list_rows.append(r)
            counter=0
            counter = collections.Counter(r)
            rows_ones_counter += counter[1] 
    return rows_ones, rows_ones_counter

def best_buddies(matrix):
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
    return best_buddies, best_buddiesQueue 

train = create_train_df ("AH171801","AH181964")
#df nn 
list_names =["par","sal1","sal2","score","tn","fn","tp","fp"]
df_nn = pd.DataFrame(columns=list_names)
#df results
list_names = ['par',"sal_1","sal_2","num_columns","num_rows","columns_1","rows_1",
              "precent_columns", "precent_rows","sum_1_colums","sum_1_rows",
              "total_numbers","total_0","1_precent","y"]#, "y"]

df_results = pd.DataFrame(columns=list_names)   
matrices = {}
flag = True
for sal_test in range(len(list_test)):
    if(sal_test>0):
        flag = True
    #2 salamanders for the test 
    sal1 = list_test[sal_test][0]
    sal2 = list_test[sal_test][1]
    
    train = create_train_df (sal1, sal2)
    test = create_test_df(sal1, sal2)
    
    for par in range(len(parameters)):
        par_name = parameters_names[par]
        new_train = remove_columns (train, parameters[par])
        new_test = remove_columns (test, parameters[par])
    #     #train
        train_y =new_train['y'] 
        train_x = new_train.drop(["num1","num2","sal1","scale","sal2",'y'], axis=1)
        train_x = np.array(train_x)
        train_y =np.array(train_y)
        train_y=train_y.astype('int')
        #test 
        #test
        test_y =new_test['y']
        test_y = test_y.astype('int')
        test_x = new_test.drop(["num1","num2","sal1","scale","sal2",'y'], axis=1)
        test_x = np.array(test_x)
        test_y =np.array(test_y)
        #nn 
        per, results ,score = nn_model(train_x, train_y, test_x, test_y)
        tn, fp, fn, tp = metrics.confusion_matrix(test_y, results).ravel()
        # ##dataframe result nn 
        
        #adding to the test dataframe the results 
        df_nn = df_nn.append({'par':par_name,
                              "sal1":sal1,
                              "sal2":sal2,
                            "score":score,
                              "tn":tn,
                            "fp":fp,
                            "tp":tp,
                            "fn":fn
                      }, ignore_index = True)
        #
        test["per"] =per[:,1]

        pairs = pd.unique(list(zip(test["sal1"],test["sal2"])))
        for pair in range(len(pairs)):
            
            couple = pairs[pair]
            new_df =test[(test["sal1"] == couple[0]) & (test["sal2"] == couple[1] )]
            #only y==1
            new_df =new_df.drop(new_df[new_df["y"] == "0"].index)
            
            empty_array = np.zeros((max(new_df["num1"]) +1, max(new_df["num2"]) +1), float)            
            list_num1 = list(new_df["num1"])
            list_num2 = list(new_df["num2"])
            list_per = list(new_df["per"])
            list_zero =list(per[:,1]) 
            list_res = list(results)
            #creating an array with probability estimates
            for i in range(len(list_num1)):
                empty_array[list_num1[i]][list_num2[i]] = list_zero[i]
            #############################################################################################3
            #Yarin 
            #empty_array is the METRIX for the best Buddies,
            #call tje function fron here 
            
            if(pair<1 and flag):
                best_buddies_list,all_best_buddies_queue = best_buddies(empty_array)
                all_best_buddies[pairs[0][0], pairs[0][1],par] = best_buddies_list
            ########################################################################################################
            key =par_name +"_"+str(couple[0])+"_"+str(couple[1])   
            matrices[key] = empty_array 
            #array with 1 and 0 
            empty_array_one = np.zeros((max(new_df["num1"]) +1, max(new_df["num2"]) +1), int) 
            
            for j in range(len(list_num1)):
                empty_array_one[list_num1[j]][list_num2[j]] = list_res[j]
                
            columns_ones, counter_ones_columns = count_one_colimns(empty_array_one)

            rows_ones, rows_ones_counter = count_one_rows(empty_array_one)
        
            # #cm          
            # y_true = list(test["y"])
            df_results = df_results.append({'par':par_name,
                                            "sal_1":couple[0],
                                            "sal_2":couple[1],
                                            "num_columns":empty_array.shape[1]-1
                                            ,"num_rows":empty_array.shape[0]-1,
                                            "columns_1":columns_ones
                                            ,"rows_1":rows_ones,
                                          "precent_columns":(columns_ones/(empty_array.shape[1]-1))*100, 
                                          "precent_rows":(rows_ones/(empty_array.shape[0]-1))*100,
                                          "sum_1_colums":counter_ones_columns,
                                          "sum_1_rows":rows_ones_counter,
                                            "total_numbers":(empty_array.shape[1]-1)*(empty_array.shape[0]-1),
                                          "total_0":(empty_array.shape[1]-1)*(empty_array.shape[0]-1)-(counter_ones_columns),
                                          "1_precent":counter_ones_columns/((empty_array.shape[1]-1)*(empty_array.shape[0]-1))*100,
                                          #"y":np.sum(test_y)
                                          #  "tn":cm[0][0],
                                          # "fp":cm[1][0],
                                          # "tp":cm[1][1],
                                          # "fn":cm[0][1],
                                          # "tn":tn,
                                          # "fp":fp,
                                          # "tp":tp,
                                          # "fn":fn,
                                           "y": np.sum(np.array(new_df["y"]))
                                          }, ignore_index = True)
            
pickle.dump( df_results, open("df_results.p", "wb" ) )
pickle.dump( df_nn, open("df_nn.p", "wb" ) )
