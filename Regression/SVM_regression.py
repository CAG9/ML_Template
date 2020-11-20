#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:31:37 2020

@author: Cesar Arcos
"""
#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import the dataset
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:,1:2].values
Y = df.iloc[:,2:3].values

# Split the dataset into training and testing set 
"""
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)
"""
# Scale the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)
# Train the model with SVR
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(X,Y)
# Predict 
y_pred = sc_Y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5]]))))
# Graphic representation of the results
X_grid = np.arange(min(X), max(X), 0.1) # to adjust the curve with more points
X_grid = X_grid.reshape(len(X_grid), 1) # convert the vector into a matrix
plt.scatter(X, Y, color = "red")
plt.plot(X_grid, regression.predict(X_grid ), color = "blue")
plt.title("SVR Regression Model")
plt.xlabel("Employee position")
plt.ylabel("Salary (USD dollar)")
plt.show()
#Extra
X_init = np.arange(sc_X.inverse_transform(min(X)),
                   sc_X.inverse_transform(max(X)),
                   0.1)
X_init = X_init.reshape(-1,1)
plt.scatter(sc_X.inverse_transform(X),
            sc_Y.inverse_transform(y),
            c='red')
plt.plot(X_init,
         sc_Y.inverse_transform(regression.predict(sc_X.transform(X_init))),
         c='blue')
plt.title('SVR Regression Model')
plt.xlabel('Employee position')
plt.ylabel('Salary (USD dollar)')
plt.show()
