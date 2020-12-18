import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2.0 # golden ratio

def golden_ratio_flower(NTURNS):
    """
    Inspired by Numberphile video on the Golden Ratio:
    https://www.youtube.com/watch?v=sj8Sg8qnjOg&t=311s

    I wrote a short script to generate a "Golden Ratio Flower" as described in the 
    Numberphile video in the youtube link.
    
    Inputs
    ------
    NTURNS: float
        number of turns for each seed placement. Can be float
    """
    theta = 360.0 / NTURNS 
    theta = theta * np.pi / 180.0
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) #rotation matrix
    x = [] #to hold the data
    y = []
    D = 1
    x.append(D) #Choose starting point to be (1,0), although theoretically
    y.append(0) #any starting point should work just as fine.
    NTRIES = int(NTURNS) * 100 #Use as many seeds as possible to make a nice dense flower.
    L = 1 #scaling factor because seeds cannot lie on top of each other.
    count = 0
    for i in range(NTRIES):
        [X,Y] = np.dot(R,np.array([x[i],y[i]]))
        count = count + 1
        if count >= NTURNS:
            count = 0
            L = L + D
            [X,Y] = np.dot(L*np.eye(2),np.array([X,Y])/np.sqrt(X**2 + Y**2))
        x.append(X)
        y.append(Y) 
    plt.plot(x,y,'k.')
    plt.axis('equal')
    plt.show()
    
golden_ratio_flower(NTURNS = np.pi)
