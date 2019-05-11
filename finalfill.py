from __future__ import print_function
from pandas import read_csv

import multiprocessing
import random
import math
import numpy as np
import pandas as pd

from random import randint
import matplotlib.pyplot as plt
dataset = read_csv('/home/aiying/Machinelearning/dataorigin.csv')

headers=list(dataset)
ds=dataset.values.tolist()
nset=[]
testcolumnameset=[]
testcolumnset=[]
indexnameset=[]
# omodset=[]


j=-1
for i in headers:
    j=j+1
    indexNames = dataset[i].index[dataset[i].apply(np.isnan)]
    if len(indexNames)==0:
         continue
    indexnameset.append(indexNames)
    perset=[]
    for i in list(indexNames):
        perset.append(ds[i])
    nset.append(perset)
    newds=dataset.drop(indexNames)
    # omodset.append(newds.values.tolist())
    testcolumnameset.append(i)
    testcolumnset.append(j)


p=0
for k in testcolumnset[0:2]:
    s0 = read_csv('/home/aiying/Machinelearning/0-10/'+str(k)+'.csv')
    j=0
    for i in indexnameset[p]:
        dataset.iloc[i,k]=s0.iloc[j,0]
        j=j+1
    p=p+1

dataset.to_csv('fillin.csv')    