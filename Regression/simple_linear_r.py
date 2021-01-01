#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 11:45:38 2020
@author: Cesar Arcos
Contact info: cesar99ag@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Import the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
# Split the dataset in training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
# Scale the variables
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
# Train the classifier
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
# Predict
y_pred = regression.predict(X_test)
# Graphic representation of the results in training set
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Salary vs Years of experience (Train set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary (USD)")
plt.show()
# Graphic representation of the results in testing set
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Salary vs Years of experience (Test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary (USD)")
plt.show()
