# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

dataset= pd.read_csv("train.csv")

X=dataset.iloc[:, 1:].values
Y=dataset.iloc[:, 0].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100,criterion="entropy")
clf.fit(X_train,Y_train)
y_pred= clf.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)

p=0
q=0
for i in range (10) :
    for j in range(10):
        if(i==j) :
            p+=cm[i][j]
        q+=cm[i][j]

print(p*100/q, "is the accuracy")       

    


