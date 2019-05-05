
#%%
# For Python 2 / 3 compatability
from __future__ import print_function
from pandas import read_csv

import numpy
dataset = read_csv('/home/aiying/machinelearning/data1.csv')

import pandas as pd
import numpy as np


import random
import numpy as np
import matplotlib.pyplot as plt

testcolumn=4

dataset.fillna(dataset.median(),inplace=True)
dataset.median()
dataset.to_csv('/home/aiying/machinelearning/addmean1.csv')
ds=dataset.values.tolist()


#%%
ds[0]


#%%
len(ds)


#%%
len(ds[0])


#%%
ds[0][139]


#%%



