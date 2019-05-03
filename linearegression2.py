import random
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# first element is 1
def generatedata(m):
    X=[]
    Y=[]
    for i in range(m):
        row=[]
        row.append(1)
        part=np.random.normal(0,1,15)
        X1,X2,X3,X4,X5,X6,X7,X8,X9,X10=part[0:10]
        X16,X17,X18,X19,X20=part[10:15]
        temp=np.random.normal(0,math.sqrt(0.1),5)

        X11=X1+X2+temp[0]
        X12=X3+X4+temp[1]
        X13=X4+X5+temp[2]
        X14=0.1*X7+temp[3]
        X15=2*X2-10+temp[4]
        row.extend(part[0:10])
        row.append(X11)
        row.append(X12)
        row.append(X13)
        row.append(X14)
        row.append(X15)
        row.extend(part[10:15])

        X.append(row)

    for j in range(len(X)):
        y=10
        Tweights=[10]
        row=X[j]
        for c in range(1,11):
            y+=0.6**c*row[c]                                                                          
            Tweights.append(0.6**c)
        y+= np.random.normal(0,math.sqrt(0.1))  
        Y.append(y)             
    return X,Y,Tweights

def calculaterror(w,X,Y):
    Terror=0
    j=0    
    for i in range(len(X)):  
        Terror = Terror+(np.dot(X[i],w)-Y[i])**2    
        j+=1
    Terror/=j
    return Terror

def calculaterror2(W,X,Y):
    Terror2=[]
    for i in range(len(W)):
        terror2=calculaterror(W[i],X,Y)
        Terror2.append(terror2)
    return Terror2
          
X,Y,Tw=generatedata(5000)
np.array(X)
np.array(Y)
w = np.linalg.lstsq(X[0:1000], Y[0:1000], rcond=None)[0]
print("leasqurew",w)
print("Truew",Tw)

def galpha():
    alpha=[1e-13]
    ratio=1.5
    temp=alpha[0]
    while temp<1e-2:
        temp=temp*ratio
        alpha.append(temp)
    return alpha

alpha=galpha()

Xtrain=[]
Ytrain=[]
for i in range(0,1000):
    x=X[i][1:21]
    y=Y[i] 
    Xtrain.append(x)
    Ytrain.append(y)

alpha_ridge = alpha
W2=[]

for i in range (len(alpha_ridge)):
    ridge= Ridge(alpha=alpha_ridge[i],normalize=True)
    ridge.fit(Xtrain,Ytrain)
    w2=list(ridge.coef_)
    b2=ridge.intercept_
    w2.insert(0,b2)
    W2.append(w2)
  
print('W2',W2)

W3=[]
alpha_lasso =  alpha        
for i in range (len(alpha_lasso)):
    lasso= Lasso(alpha=alpha_lasso[i])
    lasso.fit(Xtrain,Ytrain)
    w3=list(lasso.coef_)
    b3=lasso.intercept_
    w3.insert(0,b3)
    W3.append(w3)

print('W3',W3)    

Xtest=X[1000:5000]
Ytest=Y[1000:5000]

Terror=calculaterror(w,Xtest,Ytest)
print("Terrorlesqure",Terror)
Terror=[Terror]*len(alpha_ridge)

Terror2=calculaterror2(W2,Xtest,Ytest)
print('Terror2',Terror2)
armin=alpha_ridge[Terror2.index(min(Terror2))]
w2=W2[Terror2.index(min(Terror2))]
print('w2',w2)

# Terror2train=calculaterror2(W2,X[0:1000],Y[0:1000])
# print('Terror2train',Terror2train)
# armintrain=alpha_ridge[Terror2train.index(min(Terror2train))]

# Terror3train=calculaterror2(W3,X[0:1000],Y[0:1000])
# print('Terror3train',Terror3train)
# almintrain=alpha_lasso[Terror3train.index(min(Terror3train))]

Terror3=calculaterror2(W3,Xtest,Ytest)
print('Terror3',Terror3)
almin=alpha_lasso[Terror3.index(min(Terror3))]
w3=W3[Terror3.index(min(Terror3))]
print('w3',w3)

l0=[]
for i in range(len(w3)): 
    if abs(w3[i])<1e-10:
        l0.append(i)  
print('l0',l0)

Xtrain2=[]
Ytrain2=Y[0:1000]
for i in range(0,1000): 
    x=X[i][1:21]
    l=[n-1 for n in l0]
    for j in range(len(l0)):
        del x[l[j]]
        l=[m-1 for m in l]
    Xtrain2.append(x)

alpha_ridge = alpha
W2n=[]

for i in range (len(alpha_ridge)):
    ridge= Ridge(alpha=alpha_ridge[i],normalize=True)
    ridge.fit(Xtrain2,Ytrain2)
    w2=list(ridge.coef_)
    b2=ridge.intercept_
    w2.insert(0,b2)
    for j in range(len(l0)):
        w2.insert(l0[j],0.0)    
    W2n.append(w2)

Terror2n=calculaterror2(W2n,Xtest,Ytest)
print('Terror2n',Terror2n)
arminn=alpha_ridge[Terror2n.index(min(Terror2n))]

w2n=W2n[Terror2n.index(min(Terror2n))]
print('w2n',w2n)

print('arminn',arminn)
print('armin',armin)
print('almin',almin)

# print('armintrain',armintrain)
# print('almintrain',almintrain)

fig,ax = plt.subplots()
Y =[Terror, Terror2, Terror2n,Terror3]
ax.plot(alpha_ridge, Y[0],color='r')
ax.plot(alpha_ridge, Y[1],color='g')
ax.plot(alpha_ridge, Y[2],color='b')
ax.plot(alpha_ridge, Y[3],color='y')

plt.grid()
plt.show()












