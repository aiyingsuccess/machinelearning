
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

dataset.fillna('N',inplace=True)
dataset.median()
dataset.to_csv('/home/aiying/machinelearning/addmean1.csv')
ds=dataset.values.tolist()


#%%
header=list(dataset)


#%%
header


#%%
ds[3][4]


#%%
header


#%%
header.index('imaginedexplicit1')


#%%



