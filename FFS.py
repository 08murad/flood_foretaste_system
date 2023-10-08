#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
df1 = pd.read_csv(r'C:\Users\HP\Downloads\antarctica_mass_200204_202306 (1).csv')
df1.plot()



x=df1.drop(columns=['m','flood'])
y=df1['flood']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
p=model.predict(x_test)
accuracy_score(y_test,p)


plt.figure(figsize=(20, 10))
tree.plot_tree(model, filled=True, feature_names=x.columns)
plt.show()



