# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 23:42:43 2019

@author: 18646
"""
#Importing libraries:

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score,f1_score
from sklearn.linear_model import LogisticRegression

def count_y_class(y):
    class_0 = 0
    class_1 = 0    
    for val in y:
        if(val == 0):
            class_0 += 1
        else:
            class_1 += 1
    
    objects = ['genuine', 'forged']
    y_pos = [class_0, class_1]
    
    plt.bar(objects, y_pos, align = 'center')
    plt.xlabel('Class of Notes', fontsize=10)
    plt.ylabel('Count of classes', fontsize=10)
    #plt.xticks(y_pos, objects, fontsize=5, rotation=30)
    plt.title('Initial Plot of Data')
    plt.show()
    
    #print("Orginal class 0: ", class_0)
    #print("Orginal class 1: ", class_1)
    

def scale_data(x_train, x_test):
    scale = StandardScaler()
    x_train = scale.fit_transform(x_train)
    x_test = scale.transform(x_test)
    
    return x_train, x_test

def main():    
    col_names = ['variance', 'skew', 'curtosis', 'entropy', 'outcome']
    
    file_name = 'data_banknote_authentication.txt'
    
    file = pd.read_csv(file_name, sep=',', lineterminator='\n',skiprows = [0],
                        header = None, names = col_names)
    
    y = file.loc[:,'outcome']
    x = file.loc[:, file.columns != 'outcome']
    
    count_y_class(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 6820)
    x_train, x_test = scale_data(x_train, x_test)
    
    glm = LogisticRegression(solver='lbfgs')
    
    glm.fit(x_train, y_train)
    
    glm_predict = glm.predict(x_test)
    
    cm = confusion_matrix(y_test,glm_predict)
    
    acc = accuracy_score(y_test,glm_predict)
    
    prec = precision_score(y_test,glm_predict)
    
    recall = recall_score(y_test,glm_predict)
    
    f1 = f1_score(y_test,glm_predict)
    
    print("**********************************************************")
    print("Confusion Matrix: \n",cm)
    print("**********************************************************")
    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ",recall)
    print("F1 Score: ",f1)
    print("**********************************************************")


main()