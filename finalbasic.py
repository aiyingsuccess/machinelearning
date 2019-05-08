# For Python 2 / 3 compatability
# For Python 2 / 3 compatability
from __future__ import print_function
from pandas import read_csv

import math
import numpy as np
import pandas as pd

import random
import matplotlib.pyplot as plt
dataset = read_csv('/home/aiying/Machinelearning/dataorigin.csv')

headers = list(dataset)
ds=dataset.values.tolist()              

modset=[]
modframe=[]
modframelen=[]
testcolumnameset=[]
testcolumnset=[]
j=-1
for i in headers:
    j=j+1
    indexNames = dataset[i].index[dataset[i].apply(np.isnan)]
    if len(indexNames)==0:
        continue
    newds=dataset.drop(indexNames)
    modframe.append(newds)
    lnewds=newds.values.tolist()
    modset.append(lnewds)
    modframelen.append(len(lnewds))
    testcolumnameset.append(i)
    testcolumnset.append(j)
print('sum of columns have missing data', len(modset))
print('shortest column',min(modframelen))

with open('colmissingNO.'+'txt', 'w') as f:
        for item in testcolumnset:
            f.write("%s," % item )
with open('colmissing.'+'txt', 'w') as f:
    k=0
    for item in testcolumnset:
        f.write("%s\t" % item )
        f.write("%s\t" % headers[item])
        f.write("%s\n" % modframelen[k])
        k=k+1
