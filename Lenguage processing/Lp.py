#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:33:18 2020
@author: Cesar Arcos
Contact info: cesar99ag@gmail.com
"""
# Natural Language Processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)
# Text CLean
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
# Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
#Split the dataset into training and testing 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Train classifier naive bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predict
y_pred  = classifier.predict(X_test)
#Confusion mATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn import metrics
accuracy1 = metrics.accuracy_score(y_test, y_pred)
precision1 = metrics.precision_score(y_test, y_pred)
recall1 = metrics.recall_score(y_test, y_pred)
f1_1 = metrics.f1_score(y_test, y_pred)
#SVC
from sklearn.svm import SVC
classifier2 = SVC(kernel = "rbf", random_state = 0)
classifier2.fit(X_train, y_train)
y_pred2  = classifier2.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred2)
accuracy2 = metrics.accuracy_score(y_test, y_pred2)
precision2 = metrics.precision_score(y_test, y_pred2)
recall2 = metrics.recall_score(y_test, y_pred2)
f1_2 = metrics.f1_score(y_test, y_pred2)
# kNN
from sklearn.neighbors import KNeighborsClassifier
classifier3 = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2)
classifier3.fit(X_train, y_train)
y_pred3  = classifier3.predict(X_test)
cm3 = confusion_matrix(y_test, y_pred3)

accuracy3 = metrics.accuracy_score(y_test, y_pred3)
precision3 = metrics.precision_score(y_test, y_pred3)
recall3 = metrics.recall_score(y_test, y_pred3)
f1_3 = metrics.f1_score(y_test, y_pred3)
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier4 = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier4.fit(X_train, y_train)
y_pred4  = classifier4.predict(X_test)
cm4 = confusion_matrix(y_test, y_pred4)
accuracy4 = metrics.accuracy_score(y_test, y_pred4)
precision4 = metrics.precision_score(y_test, y_pred4)
recall4 = metrics.recall_score(y_test, y_pred4)
f1_4 = metrics.f1_score(y_test, y_pred4)
DecisionTreeClassifier()
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier5 = RandomForestClassifier(n_estimators = 100, criterion = "entropy", random_state = 0,max_features='log2')
classifier5.fit(X_train, y_train)
y_pred5  = classifier5.predict(X_test)
cm5 = confusion_matrix(y_test, y_pred5)
accuracy5 = metrics.accuracy_score(y_test, y_pred5)
precision5 = metrics.precision_score(y_test, y_pred5)
recall5 = metrics.recall_score(y_test, y_pred5)
f1_5 = metrics.f1_score(y_test, y_pred5)
#cART
from sklearn.ensemble import RandomForestClassifier
classifier6 = RandomForestClassifier(n_estimators = 100, criterion = "gini", random_state = 0,)
classifier6.fit(X_train, y_train)
y_pred6  = classifier6.predict(X_test)
cm6 = confusion_matrix(y_test, y_pred6)
accuracy6 = metrics.accuracy_score(y_test, y_pred6)
precision6 = metrics.precision_score(y_test, y_pred6)
recall6 = metrics.recall_score(y_test, y_pred6)
f1_6 = metrics.f1_score(y_test, y_pred6)
DecisionTreeClassifier()
# Naive Bayes
from sklearn.naive_bayes import CategoricalNB
classifier = CategoricalN()
classifier.fit(X_train, y_train)
# Predict
y_pred  = classifier.predict(X_test)
#Confusion MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn import metrics
accuracy1 = metrics.accuracy_score(y_test, y_pred)
precision1 = metrics.precision_score(y_test, y_pred)
recall1 = metrics.recall_score(y_test, y_pred)
f1_1 = metrics.f1_score(y_test, y_pred)
