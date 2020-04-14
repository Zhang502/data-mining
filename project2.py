#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv("iris.csv", names=names) #读取csv数据
print(dataset.describe())
data=dataset[names[:-1]]
data


# In[7]:


import numpy as np
import pandas as pd

def autoNorm(dataSet):
    norm_data=(dataSet-dataSet.min(axis=0))/(dataSet.max(axis=0)-dataSet.min(axis=0))
    return norm_data

print(autoNorm(data))


# In[9]:


def autoNorm(dataSet):
    norm_data=(dataSet-dataSet.mean(axis=0))/dataSet.std(axis=0)
    return norm_data

print(autoNorm(data))


# In[ ]:




