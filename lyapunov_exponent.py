import numpy as np
import matplotlib.pyplot as plt  # plotting libraries
import time  # time keeping tool box
from scipy import integrate
from scipy import linalg

pi = np.pi

# This code shows how to calculate the Lyapunov exponent for a nonlinear system using the equations
# proposed by:

# H. Kantz, 'A robust method to estimate the maximal Lyapunov exponent of a time series'
# Physics Letters A, Volume 185, Issue 1, Pages 77-87, 1994.

# In this code we generate the logistic map in order to create a nonlinear system for the demonstration
# of the algorithim.

def ReconSp(tauI, dm, data):
    # This function reconstructs the attractor of a time series through time delay embedding
    Nmax = len(data) - (dm - 1) * tauI
    v = np.zeros([Nmax, dm])
    for j in range(Nmax):
        for i in range(dm):
            datindex = j + i * tauI
            v[j, i] = data[datindex]
    return v

def MDneighbor(i, tahead, recsp):
    # This function calculates the nearest neighbors for some particular point along
    # the trajectory of the attractor.
    # each column in recsp must correspond to one state, each row must
    # correspond to one time step!!!!!!!!!
    # column 1 = output corresponds to distances
    # column 2 = output corresponds to index of distances
    nmax = len(recsp) - tahead
    Q1 = np.zeros([0, 2])
    for j in range(i):
        Q1 = np.vstack([Q1, [linalg.norm(recsp[i, :] - recsp[j, :]), j]])
    for j in range(i + 1, nmax):
        Q1 = np.vstack([Q1, [linalg.norm(recsp[i, :] - recsp[j, :]), j]])
    p = np.argsort(Q1[:, 0])
    Q1[:, 0] = Q1[p, 0]
    Q1[:, 1] = Q1[p, 1]
    return Q1

def logistic_map_attractor(N = 5000, A = 4, data_0 = np.pi / 10.0):
    # Create the chaotic logistic map attractor
    data = np.zeros(N)
    data[0] = data_0
    for i in range(N - 1):
        data[i + 1] = A * data[i] * (1 - data[i])    
    return data

def mean_attractor_radius(nmax, recsp):
    D = np.zeros(nmax)
    for count in range(nmax):
        D[count] = linalg.norm(recsp[count, :])
    return np.mean(D)

def nearest_neighbours(recsp, tauS = 1, tauL = 50, eps_tol = 0.001, nst = 0, ned = 999, theiler_window = 0):
    # returns list of nearest neighbours for each point in the attractor
    # tauS: initial relative time
    # tauL: final relative time
    # eps_tol: tolerance level to define nearest neighbours
    # nst: starting index for testing, PYTHON INDEX BEGINS AT 0!
    # ned: ending index for testing, -1 because of 0 indexing!
    # theiler_window: use a Theiler window to prevent autocorrelation error
    N = range(nst, ned + 1, 1)
    EPNb = [] # list to hold the results

    for k in range(len(N)):
        NbTbl = MDneighbor(N[k], tauL, recsp)
        [ndt, b] = np.shape(NbTbl)
        epTbl = np.zeros([0, 2])
        for kk in range(ndt):
            dist = NbTbl[kk, 0]
            theiler = NbTbl[kk, 1]
            if dist < eps_tol and abs(theiler - N[k]) >= theiler_window:
                epTbl = np.vstack([epTbl, NbTbl[kk, :]])
        epList = epTbl[:, 1]
        epList = np.hstack([N[k], epList])
        EPNb.append(epList)
        #print("Time step: {}. No. of NN: {}".format(k, len(epList) - 1))
    return np.array(EPNb)
  
def Stau(recsp, EPNb, tauS = 1, tauL = 50, nst = 0, ned = 999):   
    # This function calculates the stretching factor S(Tau) as proposed by
    # H. Kantz, 'A robust method to estimate the maximal Lyapunov exponent of a time series'
    # Physics Letters A, Volume 185, Issue 1, Pages 77-87, 1994.
    N = range(nst, ned + 1, 1)
    StauTbl = np.zeros([0, 2])
    for tad in range(tauS, tauL + 1):
        #print("Now at step: {}".format(tad))
        KDTbl = []
        ntcount = 0
        for it in range(len(N)):
            mnear = len(EPNb[it])
            if mnear < 2:
                continue
            aheadList = np.ones(mnear) * tad
            epNext = EPNb[it] + aheadList
            KantzDist = 0
            for k in range(1, mnear):
                KantzDist = KantzDist + linalg.norm(recsp[int(epNext[0]), :] - recsp[int(epNext[k]), :])
            KantzDist = np.log(KantzDist / (mnear - 1))
            KDTbl = np.hstack([KDTbl, KantzDist])
            ntcount = ntcount + 1
        StauList = [tad, np.sum(KDTbl) / ntcount]
        StauTbl = np.vstack([StauTbl, StauList]) 
    print("    Total points used: {}".format(ntcount))
    print("    Points without nearest neighbours: {}".format(len(N)-ntcount))
    x = StauTbl[:, 0] #time delay
    y = StauTbl[:, 1] #stretching factor
    return x, y

def demonstration():
    # this function demonstrates how to calculate the maximal Lyapunov exponent
    # using the functions above    
    STARTTIME = time.time()
    # create logistic map attractor data
    data = logistic_map_attractor(N = 5000, A = 4, data_0 = np.pi / 10.0)

    # time delayed reconstruction
    recsp = ReconSp(tauI = 1, dm = 3, data = data) 
    # columns correspond to dimensions,
    # rows correspond to time, so each row is a state vector with respect to t
    [nmax, dd] = np.shape(recsp)

    D = mean_attractor_radius(nmax, recsp)
    print("Mean attractor radius: {}".format(D))

    print("Calculating Nearest Neighbors...")
    EPNb = nearest_neighbours(recsp, tauS = 1, tauL = 20, eps_tol = 0.001, nst = 0, ned = 999, theiler_window = 0)

    print("Calculating S...")
    x, y = Stau(recsp, EPNb, tauS = 1, tauL = 20, nst = 0, ned = 999)

    m = np.polyfit(x[0:10], y[0:10], 1)
    print("Estimated maximal Lyapunov exponent: {}".format(m[0]))

    print("Time taken for computation: {:.3f} s".format(time.time() - STARTTIME))

    plt.figure(figsize=(15,5))
    plt.plot(x, y, '-o')
    plt.xlabel('Tau')
    plt.ylabel('S(Tau)')
    plt.grid('on')
    plt.show()
