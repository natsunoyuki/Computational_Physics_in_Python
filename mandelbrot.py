import numpy as np
import matplotlib.pyplot as plt
import time

#This code generates the Mandelbrot set!
def mandelbrot(num_iter=50,N=1000,X0=np.array([-2,2,-2,2]),fractal="Mandelbrot"):
    start = time.time()
    x0 = X0[0]
    x1 = X0[1]
    y0 = X0[2]
    y1 = X0[3]
    i = 1j
    
    [x,y] = np.meshgrid(np.linspace(x0,x1,N),np.linspace(y0,y1,N)*i)
    z = x + y
    c = x + y
    Q = np.zeros([N,N])
    
    for j in range(num_iter):
        index = np.abs(z) < np.inf
        Q[index] = Q[index] + 1
        #depending on the following conditions various fractal sets are produced.
        if fractal == "Julia":
            z = z ** 2 + -0.835 - 0.2321 * i
        elif fractal == "Mandelbrot":
            z = z ** 2 + c #this one produces the classical mandelbrot set.
        #z=z**3*exp(z)+0.33
        #z=z**2*exp(z)+0.33
        
    end = time.time() - start
    print("Time elapsed: {:.3f}s".format(end))
    
    plt.figure(figsize=(8,8))
    plt.pcolormesh(np.linspace(x0,x1,N),np.linspace(y0,y1,N),Q)
    plt.axis('equal')
    plt.show()

mandelbrot(fractal="Julia")
