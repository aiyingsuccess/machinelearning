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
colvalues=[]


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
    values=pd.unique(dataset[i])
    colvalues.append(values)

print('sum of columns have missing data', len(modset))
print('shortest column',min(modframelen))



i=0
with open('colmissing.'+'txt', 'w') as f:
        for item in testcolumnset:
            f.write("%s\t" % item )
            f.write("%s\t"%headers[item])
            f.write("%s\t" % modframelen[i])
            f.write("%s\n"%len(colvalues[i]))
            i=i+1
with open('numberofcolmissing.'+'txt', 'w') as f:
        for item in modframelen:
            f.write("%s," % item )


# i=-1
# colrefine=[]
# for item in modframelen:
#     i=i+1
#     if item<4800:
#         colrefine.append(testcolumnset[i])
# with open('colrefine.'+'txt', 'w') as f:
#         for item in colrefine:
#             f.write("%s," % item )    
    

