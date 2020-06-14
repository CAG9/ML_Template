#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 22:03:29 2020

@author: Cesar Arcos GOnzalez

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Position_Salaries.csv')


X = df.iloc[:,1:2].values
Y = df.iloc[:,2:3].values


"""
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)
"""

"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Adjust the dataset with lenear regression 




#predict
y_pred = regression.predict() 


#visualizate
X_grid = np.arange(min(X), max(X), 0.1) # to adjust the curve with more points
X_grid = X_grid.reshape(len(X_grid), 1) # convert the vector into a matrix
plt.scatter(X, Y, color = "red")
plt.plot(X_grid, regression.predict( ), color = "blue")
plt.title("Polynomial Regression Model")
plt.xlabel("Employee position")
plt.ylabel("Salary (USD dollar)")
plt.show()






