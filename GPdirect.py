from __future__ import division
from scipy import *
import matplotlib.pyplot as plt
from scipy import linalg

"""
This code shows how to calculate the correlation dimension of a nonlinear system based on the
equations proposed by:
Peter Grassberger and Itamar Procaccia (1983). 
"Measuring the Strangeness of Strange Attractors". Physica D: Nonlinear Phenomena. 9 (1‒2): 189‒208.
"""

def GPCdir(r,recsp):
    [nmax,dd]=shape(recsp)
    LL=zeros([nmax,nmax])
    for i in xrange(nmax):
        for j in xrange(nmax):
            LL[i,j]=linalg.norm(recsp[i,:]-recsp[j,:])
    NN=zeros(nmax)
    for i in xrange(nmax):
        nnl=LL[i,:]; nnl=nnl[i:nmax]; nnl=sort(nnl)
        k=nmax-i-1
        while nnl[k]>r:
            k=k-1
        NN[i]=k
    return sum(NN)*2/nmax**2


"""
Usage example:
r=linspace(0.01,10,1000)
C=zeros(len(r))
for q in xrange(len(r)):
    C[q]=GPCdir(r[q],75,10,data)
"""


