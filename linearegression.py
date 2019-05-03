import random
import numpy as np
import math
import matplotlib.pyplot as plt

def generatedata(m,w,b):
    X1=np.random.uniform(100,102,m)
    X2=[x-101 for x in X1]
    mu, sigma = 0, math.sqrt(0.1) # mean and standard deviation
    E= np.random.normal(mu, sigma, m)
    # E=0.0
    wx1=[x*w+b for x in X1]
    Y1=np.add(wx1,E)
    Y2=Y1

    sumofx1x1=sum([x*x for x in X1])
    sumofx1y1=sum(np.multiply(X1,Y1))
    sumofx1=sum(X1)
    sumofy1=sum(Y1)

    sumofx2y2=sum(np.multiply(X2,Y2))
    sumofx2x2=sum([x*x for x in X2])
    sumofx2=sum(X2)
    sumofy2=sum(Y2)
    
    w1=(m*sumofx1y1-sumofx1*sumofy1)/(m*sumofx1x1-sumofx1*sumofx1)
    w2=(m*sumofx2y2-sumofx2*sumofy2)/(m*sumofx2x2-sumofx2*sumofx2)
    b1=(sumofy1-w1*sumofx1)/m
    b2=(sumofy2-w2*sumofx2)/m

    return w1,w2,b1,b2


if __name__ == '__main__':
    
    W1,W2,B1,B2=[],[],[],[]
    Vw1,Vw2,Vb1,Vb2=0,0,0,0

    for i in range(1000):
        w1,w2,b1,b2=generatedata(200,1,5)
        W1.append(w1)
        W2.append(w2)
        B1.append(b1)
        B2.append(b2)

    print(len(W1))
   
    Ew1=sum(W1)/len(W1)
    Ew2=sum(W2)/len(W2)
    Eb1=sum(B1)/len(B1)
    Eb2=sum(B2)/len(B2)

    for i in range(1000): 
        Vw1+=(W1[i]-Ew1)**2
        Vw2+=(W2[i]-Ew2)**2
        Vb1+=(B1[i]-Eb1)**2
        Vb2+=(B2[i]-Eb2)**2
    
    Vw1=Vw1/1000
    Vw2=Vw2/1000
    Vb1=Vb1/1000
    Vb2=Vb2/1000

    Refw1=0.1/200*12/4
    Refw2=0.1/200*12/4
    Refb1=0.1/200*12/4*(101*101+1/12*4)
    Refb2=0.1/200*12/4*(1/12*4)
    
    print("w1",Ew1,"w2",Ew2,"b1",Eb1,"b2",Eb2)

    print("Vw1",Vw1,"Vw2",Vw2,"Vb1",Vb1,"Vb2",Vb2)
    print("Vw1",Refw1,"Vw2",Refw2,"Vb1",Refb1,"Vb2",Refb2)  
    


