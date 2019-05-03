#!/usr/bin/env python
# coding: utf-8

# In[154]:


from pandas import read_csv
import numpy
dataset = read_csv('/home/aiying/Desktop/data.csv')


# In[155]:


import pandas as pd
import numpy as np


# In[160]:


s=dataset.isnull().sum()


# In[161]:


s.sort_values(ascending=True)


# In[ ]:





# In[145]:


dataset.mean()


# In[149]:


dataset.fillna(dataset.mean(),inplace=True)


# In[151]:


dataset.to_csv('addmean.csv')


# In[152]:


print(dataset.isnull().sum())


# In[ ]:




