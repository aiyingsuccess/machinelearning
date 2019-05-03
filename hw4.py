import numpy as np
import random
from scipy.stats import norm
from scipy.stats import chi2
from numpy import linalg as LA
from operator import add
import matplotlib.pyplot as plt
from functools import reduce


def generatedata(m,k,e):
    X=[]
    Y=[]
    for i in range(m):
        row_normal = list(norm.rvs(size=k-1,loc=0,scale=1))  
        temp=e+float(1/2*chi2.rvs(df=2,size=1)) #1/2 chisquare
        seed=random.random()
        if(seed<0.5):
            xk=temp
        else:
            xk=-temp
        row_normal.append(xk)
        if xk>0:
            y=1
        else:
            y=-1
        X.append(row_normal)  
            
        Y.append(y)
    return X,Y

def learningPerceptron(X,Y):
    m=len(X)
    k=len(X[0])
    w=[0]*(k+1)
    
    for i in range(m):
        X[i].insert(0,1)
        for j in range(len(X[i])):
            scale=LA.norm(X[i],2)
            X[i][j]=X[i][j]/scale
    index=0
    right=0
    step=0
    Wlist=[]
    while 1:
        if np.dot(X[index],w)*Y[index]<=0:
            w=list(map(add, w, np.dot(X[index],Y[index])))
            step=step+1
            print("modify w",w)
            Wlist.append(w)
            if step>10*m:
                break

        elif np.dot(X[index],w)*Y[index]>0:                        #elif!not if
            # print(">0",np.dot(X[index],w),"index",index)    
            right=right+1
            if right==m:
                break

        index=index+1        
        if index==m:
            index=0
            right=0          
    return w,step,Wlist

def q2():
    X,Y=generatedata(100,20,1)
    W,step,Wlist=learningPerceptron(X,Y)
    print("W",W)
    print("Timestep",step)

def q3(repeat):
    Step=[]
    E=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for i in range(len(E)):
        e=E[i]
        step=0
        for j in range(repeat):
            X,Y=generatedata(100,20,e)
            W,s,Wlist=learningPerceptron(X,Y)
            step=step+s
        step=step/repeat
        Step.append(step)
    plt.plot(E,Step)
    plt.grid()
    plt.show()

def q4(repeat):
    Step=[]
    K=range(2,42,2)
    for i in range(len(K)):
        k=K[i]
        step=0
        for j in range(repeat):
            X,Y=generatedata(1000,k,1)
            W,s,Wlist=learningPerceptron(X,Y)
            step=step+s
        step=step/repeat
        Step.append(step)
    plt.plot(K,Step)
    plt.grid()
    plt.show()

def f(L):
    def add(x,y):
        return x+y
    return reduce(add,[x*x for x in L])

def generatedata2(m,k):
    X=[]
    Y=[]
    for i in range(m):
        row = list(norm.rvs(size=k,loc=0,scale=1))  
        X.append(row)
    for j in range(len(X)):
        if f(X[j])>=k:
            y=1
        else:
            y=-1        
        Y.append(y)
    return X,Y

def q5():
    X,Y=generatedata2(100,2)
    plt.plot([x[0] for x in X if Y[X.index(x)]==1],[x[1] for x in X if Y[X.index(x)]==1],'ro')
    plt.plot([x[0] for x in X if Y[X.index(x)]==-1],[x[1] for x in X if Y[X.index(x)]==-1],'bo')
    plt.show()
    
    W,s,Wlist=learningPerceptron(X,Y)
    plt.plot([w[1] for w in Wlist],[w[2] for w in Wlist],'bo')
    plt.show()
             
if __name__ == '__main__':
    # q2()
    # q3(30)
    # q4(50)
    q5()
    

    

    

