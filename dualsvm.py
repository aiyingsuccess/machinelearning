import numpy as np
from numpy import linalg as LA
import math
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sympy import *

global e             
e=1
global step                  
step=0.01

def kernel(x,y):
    return (1+np.dot(x,y))**2

def solver1(alphas,a):
    x=[[1,1],[1,-1],[-1,-1],[-1,1]]
    y=[-1,1,-1,1]
    m=4
    sum2=sum(alphas)
    t=symbols('t')
    sum1=exp(t)
    sum5=cos(t)
    sum6=cos(t)
    p1=cos(t)
    for i in range(0,m-1):
        p1=p1+ln(alphas[i])
        sum1-=alphas[i]*y[i]*y[3]
        if kernel(x[i],x[3])!=0:
            print("i am here",i)
        sum5+=alphas[i]*y[i]*kernel(x[i],x[3])
        for j in range(0,m-1):
            sum6+=alphas[i]*y[i]*kernel(x[i],x[j])*y[j]*alphas[j]
    
    sum1=sum1-exp(t)
    print('sum1',sum1)
    sum5=sum5-cos(t)
    sum6=sum6-cos(t)
    p1=p1-cos(t)

    sum3=sum1**2*kernel(x[3],x[3])
    sum4=sum1*2*y[3]
    
    print(sum1,'',sum2,'',sum3,'',sum4,'',sum5,'',sum6)
    f=-(sum1+sum2-1/2*(sum3+sum4*sum5+sum6))
    
    while 1:
        global e
        if(e>1e-30):
           e=e/2
        print("e",e)
      
        F=f-e*(p1+ln(sum1))
        sg = [-diff(F,alphas[0]),-diff(F,alphas[1]),-diff(F,alphas[2])]
        sgvalue=[float(sg[0].subs([(alphas[0],a[0]),(alphas[1],a[1]),(alphas[2],a[2])])),
        float(sg[1].subs([(alphas[0],a[0]),(alphas[1],a[1]),(alphas[2],a[2])])),
        float(sg[2].subs([(alphas[0],a[0]),(alphas[1],a[1]),(alphas[2],a[2])]))]
        
        global step
        an=np.add(a,np.multiply(sgvalue,step))
        dif=list(np.subtract(a,an))
        Fvalue=float(F.subs([(alphas[0],a[0]),(alphas[1],a[1]),(alphas[2],a[2])]))
        print('F',F)
        print('Fvalue',Fvalue)
        if LA.norm(dif)<1e-30:
            break
        
        a=an
        print(a)               # update of alpha1, alpha2, alpha3
    sum1value=float(sum1.subs([(alphas[0],a[0]),(alphas[1],a[1]),(alphas[2],a[2])]))
    return a,sum1value


alphas=['al0','al1','al2']   
alphas[0],alphas[1],alphas[2]=symbols('al0,al1,al2')
a=[0.7,0.05,0.1]                         #initial assignment of alpha
a,alpha4=solver1(alphas,a) 
print('alpha4',alpha4)
m=list(a)
m.append(alpha4)
print('final alphas',m)

