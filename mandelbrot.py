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
