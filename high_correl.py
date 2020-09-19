import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time

starttime=time.time()

# The functions in this script are based on the paper:
# A M Albano, A Passamante, M E Farrell, Physica D 54 (1991) 85-97
# to calculate high order auto-correlations of a time series.
# Used in non-linear time series analysis to determine the lag time
# for use in time-delay attractor reconstruction.

def MultMoment(tauI,mulist,xlist):
    # A M Albano, A Passamante, M E Farrell, Physica D 54 (1991) 85-97
    avx = np.mean(xlist) #mean of data
    xdouble = np.power(xlist, 2) #data squared
    avdouble = np.mean(xdouble) #mean of data squared
    xstd = np.sqrt(avdouble - avx ** 2) #standard deviation of data
    dm = len(mulist)
    di = sum(mulist)
    nmax = len(xlist) - (dm - 1) * tauI
    
    def elempol(tauI, i):
        a = 1
        
        for j in range(dm):
            a = a * (xlist[i + (j) * tauI] - avx) ** mulist[j]
            
        return a
    
    def pol(tauI):
        a = 0
        
        for i in range(nmax):
            a = a + elempol(tauI, i)
            
        return a
    
    mulmo = pol(tauI) / xstd ** di / nmax
    return mulmo

def High2Corr(tauI, xlist):
    mul = np.array([1, 1])
    kap = MultMoment(tauI, mul, xlist)
    return kap

def High3Corr(tauI, xlist):
    mul = np.array([1, 1, 1])
    kap = MultMoment(tauI, mul, xlist)
    return kap

def High4Corr(tauI, xlist):
    # modified
    mul1 = np.array([[1,1,0,0], [1,0,1,0], [1,0,0,1]])
    mul2 = np.array([[0,0,1,1], [0,1,0,1], [0,1,1,0]])
    kap = MultMoment(tauI, np.array([1,1,1,1,]), xlist)
    
    for i in range(3):
        kap = kap - MultMoment(tauI, mul1[i], xlist) * MultMoment(tauI, mul2[i], xlist)
        
    return kap
"""
def High4Corr(tauI,xlist):
    #original
    mul=sorted(list(set(list(itertools.permutations([1,1,0,0])))))
    mul=flipud(mul)
    kap=MultMoment(tauI,array([1,1,1,1,]),xlist)
    j=0
    for i in range(3):
        j=j-1
        kap=kap-MultMoment(tauI,mul[i],xlist)*MultMoment(tauI,mul[j],xlist)
    return kap
"""
def High5Corr(tauI, xlist):
    # modified
    mul1 = np.array([[1,1,1,0,0],
                     [1,1,0,1,0],
                     [1,1,0,0,1],
                     [1,0,1,1,0],
                     [1,0,1,0,1],
                     [1,0,0,1,1],
                     [0,1,1,1,0],
                     [0,1,1,0,1],
                     [0,1,0,1,1],
                     [0,0,1,1,1]])
    
    mul2 = np.array([[0,0,0,1,1],
                     [0,0,1,0,1],
                     [0,0,1,1,0],
                     [0,1,0,0,1],
                     [0,1,0,1,0],
                     [0,1,1,0,0],
                     [1,0,0,0,1],
                     [1,0,0,1,0],
                     [1,0,1,0,0],
                     [1,1,0,0,0]])
    kap = MultMoment(tauI, np.array([1,1,1,1,1]), xlist)
    
    for i in range(10):
        kap = kap - MultMoment(tauI, mul1[i], xlist) * MultMoment(tauI, mul2[i], xlist)
        
    return kap
"""
def High5Corr(tauI,xlist):
    #original
    mul3=sorted(list(set(list(itertools.permutations([1,1,1,0,0])))))
    mul3=flipud(mul3)
    mul2=sorted(list(set(list(itertools.permutations([1,1,0,0,0])))))
    mul2=flipud(mul2)
    kap=MultMoment(tauI,array([1,1,1,1,1]),xlist)
    j=0
    for i in range(10):
        j=j-1
        kap=kap-MultMoment(tauI,mul3[i],xlist)*MultMoment(tauI,mul2[j],xlist)
    return kap
"""
def High6Corr(tauI, xlist):
    # mul4=sorted(list(set(list(itertools.permutations([1,1,1,1,0,0])))))
    # mul4=flipud(mul4)
    # mul2=sorted(list(set(list(itertools.permutations([1,1,0,0,0,0])))))
    # mul2=flipud(mul2)
    mulist6s1 = np.array([[0, 0, 1, 1, 1, 1],
                          [0, 1, 0, 1, 1, 1],
                          [0, 1, 1, 0, 1, 1],
                          [0, 1, 1, 1, 0, 1],
                          [0, 1, 1, 1, 1, 0],
                          [1, 0, 0, 0, 1, 1],
                          [1, 0, 0, 1, 0, 1],
                          [1, 0, 0, 1, 1, 0],
                          [1, 0, 0, 1, 1, 1],
                          [1, 0, 1, 0, 0, 1],
                          [1, 0, 1, 0, 1, 0],
                          [1, 0, 1, 0, 1, 1],
                          [1, 0, 1, 1, 0, 0],
                          [1, 0, 1, 1, 0, 1],
                          [1, 0, 1, 1, 1, 0],
                          [1, 1, 0, 0, 0, 1],
                          [1, 1, 0, 0, 1, 0],
                          [1, 1, 0, 0, 1, 1],
                          [1, 1, 0, 1, 0, 0],
                          [1, 1, 0, 1, 0, 1],
                          [1, 1, 0, 1, 1, 0],
                          [1, 1, 1, 0, 0, 0],
                          [1, 1, 1, 0, 0, 1],
                          [1, 1, 1, 0, 1, 0],
                          [1, 1, 1, 1, 0, 0]])
    
    mulist6s2 = np.array([[1, 1, 0, 0, 0, 0],
                          [1, 0, 1, 0, 0, 0],
                          [1, 0, 0, 1, 0, 0],
                          [1, 0, 0, 0, 1, 0],
                          [1, 0, 0, 0, 0, 1],
                          [0, 1, 1, 1, 0, 0],
                          [0, 1, 1, 0, 1, 0],
                          [0, 1, 1, 0, 0, 1],
                          [0, 1, 1, 0, 0, 0],
                          [0, 1, 0, 1, 1, 0],
                          [0, 1, 0, 1, 0, 1],
                          [0, 1, 0, 1, 0, 0],
                          [0, 1, 0, 0, 1, 1],
                          [0, 1, 0, 0, 1, 0],
                          [0, 1, 0, 0, 0, 1],
                          [0, 0, 1, 1, 1, 0],
                          [0, 0, 1, 1, 0, 1],
                          [0, 0, 1, 1, 0, 0],
                          [0, 0, 1, 0, 1, 1],
                          [0, 0, 1, 0, 1, 0],
                          [0, 0, 1, 0, 0, 1],
                          [0, 0, 0, 1, 1, 1],
                          [0, 0, 0, 1, 1, 0],
                          [0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 1, 1]])
    
    mulist6p1 = np.array([[1, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 1, 0],
                          [1, 0, 0, 0, 1, 0],
                          [1, 0, 0, 0, 1, 0],
                          [1, 0, 0, 1, 0, 0],
                          [1, 0, 0, 1, 0, 0],
                          [1, 0, 0, 1, 0, 0],
                          [1, 0, 1, 0, 0, 0],
                          [1, 0, 1, 0, 0, 0],
                          [1, 0, 1, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0]])
    
    mulist6p2 = np.array([[0, 1, 0, 0, 1, 0],
                          [0, 1, 0, 1, 0, 0],
                          [0, 1, 1, 0, 0, 0],
                          [0, 1, 0, 0, 0, 1],
                          [0, 1, 0, 1, 0, 0],
                          [0, 1, 1, 0, 0, 0],
                          [0, 1, 0, 0, 0, 1],
                          [0, 1, 0, 0, 1, 0],
                          [0, 1, 1, 0, 0, 0],
                          [0, 1, 0, 0, 0, 1],
                          [0, 1, 0, 0, 1, 0],
                          [0, 1, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0, 1],
                          [0, 0, 1, 0, 1, 0],
                          [0, 0, 1, 1, 0, 0]])
    
    kap = MultMoment(tauI, np.array([1,1,1,1,1,1]), xlist)
    
    for i in range(25):
        kap = kap - MultMoment(tauI, mulist6s1[i], xlist) * MultMoment(tauI, mulist6s2[i], xlist)
        
    for i in range(15):
        kap = kap + MultMoment(tauI, mulist6p1[i], xlist) * MultMoment(tauI, mulist6p2[i], xlist)*2
        
    return kap

def rossler_attractor(x, t):
    a = 0.2
    b = 0.4
    c = 5.7
    xdot = np.zeros(len(x))
    xdot[0] = -x[1] - x[2]  # x
    xdot[1] = x[0] + a * x[1]  # y
    xdot[2] = b + x[2] * (x[0] - c)  # z
    return xdot
    
def high_correl_demonstration():
    # This function shows how to use the functions above to calculate higher
    # order correlations
    dt = np.pi / 100.0
    t = np.arange(0, dt*(1048576+1000), dt)
    x0 = np.array([10, 0, 0])
    a = integrate.odeint(rossler_attractor, x0, t)
    X = a[:,0]
    X = X[1000000:]
    
    T = np.arange(1, 200+1, 1)
    C2 = np.zeros(len(T))
    #C3 = np.zeros(len(T))
    #C4 = np.zeros(len(T))
    #C5 = np.zeros(len(T))
    #C6 = np.zeros(len(T))
    for i in range(len(T)):
        C2[i] = High2Corr(T[i], X)
        #C3[i]=High3Corr(T[i],X)
        #C4[i]=High4Corr(T[i],X)
        #C5[i]=High5Corr(T[i],X)
        #C6[i]=High6Corr(T[i],X)
    
    endtime=time.time()-starttime
    print("Time elapsed: {}".format(endtime))
    
    #plt.plot(T,C2,'b',T,C3,'r',T,C4,'g',T,C5,'y',T,C6,'k')
    #plt.legend(['C2','C3','C4','C5','C6'])
    plt.plot(T, C2)
    plt.grid('on')
    #plt.axis([0,1000,-0.1,0.2])
    plt.plot(T, T *0 + 1 / np.exp(1),'r-.')
    plt.plot(T, T * 0 + 1 - 1 / np.exp(1),'r-.')
    plt.plot(T,T * 0,'r-.')
    plt.show()

