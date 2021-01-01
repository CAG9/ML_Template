#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:51:03 2020
@author: Cesar Arcos
Contact info: cesar99ag@gmail.com
"""
#Apriori
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Import dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)]) 
# Training the apriori algorithm
from apyori import apriori
rules = apriori(transactions, min_support = 0.003 , min_confidence = 0.2,
                min_lift = 3, min_length = 2)
# Visualizing the results
results = list(rules)
results[4]
