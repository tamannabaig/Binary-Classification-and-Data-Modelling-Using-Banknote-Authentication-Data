# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:34:48 2019

@author: Tamanna Baig

"""
#Importing libraries:

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score,f1_score
from sklearn.neighbors import KNeighborsClassifier

   
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

def choose_k_values(x_train, x_test, y_train, y_test):
    k_val = np.arange(1, 21)
    train_accuracy = []
    test_accuracy = []
    
    for i, k in enumerate(k_val):
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(x_train, y_train)
        train_accuracy.append(knn.score(x_train,y_train))
        test_accuracy.append(knn.score(x_test,y_test))
    
    best_k = 1+test_accuracy.index(np.max(test_accuracy))
    plt.figure(figsize=(13,8))
    plt.plot(k_val, test_accuracy, label = 'Testing Accuracy')
    plt.plot(k_val, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.title('-value vs. Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.xticks(k_val)
    plt.savefig('graph.png')
    plt.show()
    
    print("-----------------------------------------------------------------")
    print('Best Accuracy is {} with K = {}'.format(np.max(test_accuracy),best_k))
    print("-----------------------------------------------------------------\n")
    return best_k
    
def calling_knn(k, x_train, x_test, y_train, y_test):
    
    knn = KNeighborsClassifier(n_neighbors = k)
    
    knn.fit(x_train, y_train)
    prediction = knn.predict(x_test)
    
    cm = confusion_matrix(y_test,prediction)
    
    acc = accuracy_score(y_test,prediction)
    
    prec = precision_score(y_test,prediction)
    
    recall = recall_score(y_test,prediction)
    
    f1 = f1_score(y_test,prediction)
    
    print("For value k: ",k)
    print("**********************************************************")
    print("Confusion Matrix: \n",cm)
    print("**********************************************************")
    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ",recall)
    print("F1 Score: ",f1)
    print("**********************************************************")

def main():    
    col_names = ['variance', 'skew', 'curtosis', 'entropy', 'outcome']
    
    file_name = 'data_banknote_authentication.txt'
    
    file = pd.read_csv(file_name, sep=',', lineterminator='\n',skiprows = [0],
                        header = None, names = col_names)
    
    y = file.loc[:,'outcome']
    x = file.loc[:, file.columns != 'outcome']
    
    #To plot the Initial Data points from the input file
    count_y_class(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 6820)
    x_train, x_test = scale_data(x_train, x_test)
    
    best_k = choose_k_values(x_train, x_test, y_train, y_test)
    
    print("Results with k value from user-designed algorithm: ")
    calling_knn(7, x_train, x_test, y_train, y_test)
    
    print("Results from best k value using SkLearn toolkit: ")
    calling_knn(best_k, x_train, x_test, y_train, y_test)
    
    print("**********************************************************")
    print("len y_test: ", len(y_test))
   
    
main()

