import numpy as np
from scipy import linalg

"""
This code shows how to calculate the correlation dimension of a nonlinear system based on the
equations proposed by:
Peter Grassberger and Itamar Procaccia (1983). 
"Measuring the Strangeness of Strange Attractors". Physica D: Nonlinear Phenomena. 9 (1‒2): 189‒208.
"""

def GPCdir(r,recsp):
    [nmax,dd] = np.shape(recsp)
    LL = np.zeros([nmax,nmax])
    for i in range(nmax):
        for j in range(nmax):
            LL[i,j] = linalg.norm(recsp[i,:] - recsp[j,:])
    NN = np.zeros(nmax)
    for i in range(nmax):
        nnl = LL[i,:] 
        nnl = nnl[i:nmax] 
        nnl = np.sort(nnl)
        k = nmax - i - 1
        while nnl[k] > r:
            k = k - 1
        NN[i] = k
    return np.sum(NN) * 2 / nmax ** 2


"""
Usage example:
r = np.linspace(0.01,10,1000)
C = np.zeros(len(r))
for q in range(len(r)):
    C[q]=GPCdir(r[q],75,10,data)
"""


