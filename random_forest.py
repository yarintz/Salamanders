# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 01:01:58 2022

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
import matplotlib.pyplot as plt
import collections

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

#
results = pickle.load( open( "df_results.p", "rb" ) )
results['y'] = (results['y'] > 0).astype(int)

parameters_names = ["division_df","subtraction_df","division_subtraction_df",
              "division_nn_df","subtraction_nn_df","division_subtraction_nn_df",
              "division_12_df","subtraction_12_df","division_subtraction_12_df",
              "division_21_df","subtraction_21_df","division_subtraction_21_df"]
random_forest_result = {}
#the model 
clf = RandomForestClassifier(max_depth=2, random_state=0)

def random_forst_cv(data,label,par):
    skf = StratifiedKFold(7, shuffle=True, random_state=1)
    cv_results = cross_validate(clf, data,label, cv = skf)   
    sorted(cv_results.keys())
    cv_results['test_score']
    print("---------------------------------------")
    print(par)
    print(cv_results['test_score'])
    random_forest_result[par] = cv_results['test_score']

for i  in range(len(parameters_names)):
     data =results[(results["par"] == parameters_names[i])]
     par = parameters_names[i]
     #remove cols
     label = data["y"]
     label = np.array(label)
     data = data.drop(["par","sal_1","sal_2","y"], axis=1)
     data = np.array(data)
     random_forst_cv(data,label,par)
     
pickle.dump( random_forest_result, open("random_forest_result.p", "wb" ) )
     