from scipy import *
import matplotlib.pyplot as plt

"""
Inspired by Numberphile video on the Golden Ratio:
https://www.youtube.com/watch?v=sj8Sg8qnjOg&t=311s

I wrote a short script to generate a "Golden Ratio Flower" as described in the 
Numberphile video in the youtube link.
"""

phi=(1+sqrt(5))/2.0

NTURNS=pi #number of turns for each seed placement
theta=360.0/NTURNS 
theta=theta*pi/180.0
R=array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]]) #rotation matrix
x=[] #to hold the data
y=[]
D=1
x.append(D) #Choose starting point to be (1,0), although theoretically
y.append(0) #any starting point should work just as fine.
NTRIES=int(NTURNS)*100 #Use as many seeds as possible to make a nice dense flower.
L=1 #scaling factor because seeds cannot lie on top of each other.
count=0
for i in range(NTRIES):
    [X,Y]=dot(R,array([x[i],y[i]]))
    count=count+1
    if count>=NTURNS:
        count=0
        L=L+D
        [X,Y]=dot(L*eye(2),array([X,Y])/sqrt(X**2+Y**2))
    x.append(X)
    y.append(Y) 
plt.plot(x,y,'k.')
plt.axis('equal')
plt.show()