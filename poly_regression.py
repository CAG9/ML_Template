#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Position_Salaries.csv')


X = df.iloc[:,1:2].values
Y = df.iloc[:,2:3].values

# Adjust the dataset with lenear regression 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)



#adjust the daset with polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X) # create a "dataset" with diferen degrees
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)

#Visualize the result with Polynomical Regression
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Linear Regression Model")
plt.xlabel("Employee position")
plt.ylabel("Salary (USD dollar)")
plt.show()

#Visualize the result with Polynomical Regression
X_grid = np.arange(min(X), max(X), 0.1) # to adjust the curve with more points
X_grid = X_grid.reshape(len(X_grid), 1) # convert the vector into a matrix
plt.scatter(X, Y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Polynomial Regression Model")
plt.xlabel("Employee position")
plt.ylabel("Salary (USD dollar)")
plt.show()

#predicting the salary of an employee
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))





