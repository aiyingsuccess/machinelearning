#!/usr/bin/env python
# coding: utf-8

# In[154]:


from pandas import read_csv
import numpy
dataset = read_csv('/home/aiying//data1.csv',low_memory=False)


# In[155]:


import pandas as pd
import numpy as np


# In[160]:


s=dataset.isnull().sum()


# In[161]:


s.sort_values(ascending=True)


# In[ ]:





# In[145]:


dataset.median()


# In[149]:


dataset.fillna(dataset.median(),inplace=True)


# In[151]:


dataset.to_csv('addmean1.csv')


# In[152]:


print(dataset.isnull().sum())


# In[ ]:




