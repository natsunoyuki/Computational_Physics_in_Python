import numpy as np
import matplotlib.pyplot as plt
from scipy import random
import time

#This code demonstrates the recursive method used to create a fractal. In this case
#the fractal is the Barnsley fern.

def fern(N=10000):
    #N is the total number of points to use. 
    #More points mean more minute detail but longer run times.
    start = time.time()
    x = np.zeros([N + 1, 2])
    x[0, :] = np.array([0.5, 0.5])
    for k in range(N):
        r = random.random()
        if r <= 0.01:
            x[k + 1, 0] = 0
            x[k + 1, 1] = 0.16 * x[k, 1]
        elif r <= 0.85:
            x[k + 1, 0] = 0.85 * x[k, 0] + 0.04 * x[k, 1]
            x[k + 1, 1] = -0.04 * x[k, 0] + 0.85 * x[k, 1] + 1.6
        elif r <= 0.93:
            x[k + 1, 0] = 0.2 * x[k, 0] - 0.26 * x[k, 1]
            x[k + 1, 1] = 0.23 * x[k, 0] + 0.22 * x[k, 1] + 1.6
        else:
            x[k + 1, 0] = -0.15 * x[k, 0] + 0.28 * x[k, 1] + 0.26
            x[k + 1, 1] = 0.26 * x[k, 0] + 0.24 * x[k, 1] + 0.44

    end = time.time() - start
    print('The total time elapsed is: {:.2f}s'.format(end))
    return x

#example function call and plotting:
x=fern(10000)
plt.figure(figsize=[8,8])
plt.plot(x[:,0], x[:,1], 'g.')
plt.axis('equal')
plt.show()
