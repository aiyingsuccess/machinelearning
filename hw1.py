import numpy as np
from math import pow
import random
import matplotlib.pyplot as plt

Lmle=[]
Lmom=[]
Emle=[]
Emom=[]
L=10

for j in range(1000):
    s=np.random.uniform(0,10,100)
    mom=2*np.mean(s)
    mle=np.max(s)
    Lmom.append(mom)
    Lmle.append(mle)
    Emom.append(pow(mom-L,2))
    Emle.append(pow(mle-L,2))
    
# print(Lmom)
# print(Lmle)

# print("MSE of MOM",sum(Emom)*0.001)
# print("MSE of MLE",sum(Emle)*0.001)

def foo(s):
    return 10 / int(s)

def bar(s):
    return foo(s) * 2

def main():
    try:
        bar('0')
    except Exception as e:
        print('Error:', e)
    finally:
        print('finally...')

main()

from functools import reduce
def f():
    def add(x,y):
        return x+y
    return reduce(add,[x*x for x in [1,3,5,7]])
print(f())


