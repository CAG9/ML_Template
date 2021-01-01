#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:05:51 2020
@author: Cesar Arcos
Contact info: cesar99ag@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values
# Train the classifier
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators = 300, random_state = 0)
regression.fit(X,Y)
# Predict
y_pred = regression.predict(6.5)
# Graphic representation of the results 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X, regression.predict(X), color = "blue")
plt.title("Random Frorest Model")
plt.xlabel("Employee position")
plt.ylabel("Salaty (USD dollar)")
plt.show()
