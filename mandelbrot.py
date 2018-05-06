# file name: mandelbrot.py

from scipy import *
import matplotlib.pyplot as plt
import time

#This code generates the Mandelbrot set!

num=50
N=1000 #size of grid
start=time.time()
X0=array([-2,2,-2,2])
x0=X0[0]
x1=X0[1]
y0=X0[2]
y1=X0[3]
i=1j

[x,y]=meshgrid(linspace(x0,x1,N),linspace(y0,y1,N)*i)
z=x+y
c=x+y
Q=zeros([N,N])

for j in range(num):
    index=abs(z)<inf
    Q[index]=Q[index]+1
    #z=z**2+-0.835-0.2321*i
    z=z**2+c
    #z=z**3*exp(z)+0.33
    #z=z**2*exp(z)+0.33
    
end=time.time()-start
print(end)
plt.pcolormesh(linspace(x0,x1,N),linspace(y0,y1,N),Q)
plt.axis('equal')
plt.axes()
plt.show()

#legacy code:
"""
N=1000 #trials

start = time.time()
# set axes:
x0 = -2
x1 = 2
y0 = -2
y1 = 2
# number of points per axis:
# num=400
num = N
# create 2D space:
X = linspace(x0,x1,num)
Y = linspace(y0,y1,num)*1j
[x,y] = meshgrid(X,Y)
# print(shape(x))
# print(shape(y))
c = x+y
z = zeros([num,num])

for j in range(num):
    z = z**2+c
    
end = time.time()-start
print("Total run time: ")
print(end)

plt.plot(real(logical_not(isnan(z))*c),imag(logical_not(isnan(z))*c),'k.',markersize=1)
# plt.imshow(abs(z))
plt.axis('equal')
plt.show()
"""
