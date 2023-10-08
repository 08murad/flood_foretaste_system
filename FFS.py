#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[142]:


import pandas as pd

df = pd.read_csv(r'C:\Users\HP\Downloads\All-floods-in-Bangladesh_Dataset.csv')

df.dropna()
s=df['m']
b=df['d']
for i in range(0,99):
    t=(30*s[i]+b[i])
    print(t)


# In[ ]:





# In[59]:


plt.show()


# In[81]:


df1 = pd.read_csv(r'C:\Users\HP\Downloads\antarctica_mass_200204_202306 (1).csv')
df1.plot()


# In[133]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
df1 = pd.read_csv(r'C:\Users\HP\Downloads\antarctica_mass_200204_202306 (1).csv')
df1.plot()


# In[132]:


x=df1.drop(columns=['m','flood'])
y=df1['flood']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
p=model.predict(x_test)
accuracy_score(y_test,p)


# In[ ]:





# In[ ]:




