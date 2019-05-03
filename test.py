import numpy as np
from scipy.optimize import minimize
from sympy import *
import matplotlib.pyplot as plt

n1 = [0 for i in range(15)]
n2 = [0] * 15

print(n1)
print(n2)

n1[0:11] = [10] * 10

print(n1)

from numpy import linalg as LA
a = [0,0,10]
print(LA.norm(a,2))


print(a)
a.append(n1[0:2])
print(type(a))    

print(list(range(1,10)))
x=2
x+=2
print(x)

l1=[1,2,3]
l2=[l1[1]* i for i in l1]
print(sum(l2))


def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen, x0, method='nelder-mead',
               options={'xtol': 1e-8, 'disp': True})

print(res.x)

a=['x','y']
a[0],a[1] = symbols('a0,a1')

s = a[0]**2+a[1]**2+a[0]*a[1]
s=sum(a)-a[1]
ss=symbols('0')
print(s+ss)
sprime = -diff(s,a[0])
value=sprime.subs(a[0],1)
print(float(value))
print((s).evalf(subs={a[0]:1.0}))

a =[1,2]
c=np.multiply(a,2)
print(np.add(a,c))
b=[1,2]
print('index',b.index(min(b)))
b=c
print(b)
a.append(3)
print(b)

print(b[0:2])

#First create some toy data:
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

#Creates just a figure and only one subplot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

#Creates two subplots and unpacks the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)

#Creates four polar axes, and accesses them through the returned array
fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
axes[0, 0].plot(x, y)
axes[1, 1].scatter(x, y)

#Share a X axis with each column of subplots
plt.subplots(2, 2, sharex='col')

#Share a Y axis with each row of subplots
plt.subplots(2, 2, sharey='row')

#Share both X and Y axes with all subplots
plt.subplots(2, 2, sharex='all', sharey='all')

#Note that this is the same as
plt.subplots(2, 2, sharex=True, sharey=True)

#Creates figure number 10 with a single subplot
#and clears it if it already exists.
fig, ax=plt.subplots(num=10, clear=True)

plt.show()

m=[1,2] 
a=[3,4]
def tes(m):
    m=a
    print('m1',m)
def tes2(m):
    m[0]=a[0]
    print('m3',m)
tes(m)
print('m2',m)
tes2(m)
print('m4',m)