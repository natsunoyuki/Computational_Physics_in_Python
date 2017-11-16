#This is a placeholder file to hold useful bits of code which are used often    
#for a short period of time. Do not store important stuff here!!!!!!!!!!!!!!
#for reference this line is exactly 80 characters long#########################
from __future__ import division  # float division
from scipy import *  # generic math libraries import
import matplotlib.pyplot as plt  # plotting libraries
import time  # time keeping tool box
from scipy import integrate
from scipy import linalg

#This code shows how to calculate the Lyapunov exponent for a nonlinear system using the equations
#proposed by:

#Holger Kantz, 'A robust method to estimate the maximal Lyapunov exponent of a time series'
#Physics Letters A, Volume 185, Issue 1, Pages 77-87, 1994.

def ReconSp(tauI,dm,data):
    #this function reconstructs the attractor of a time series through time delay embedding
    Nmax=len(data)-(dm-1)*tauI
    v=zeros([Nmax,dm])
    for j in range(Nmax):
        for i in range(dm):
            datindex=j+i*tauI
            v[j,i]=data[datindex]
    return v

def MDneighbor(i,tahead,recsp):
    #This function calculates the nearest neighbors for some particular point along
    #the trajectory of the attractor.
    #each column in recsp must correspond to one state, each row must
    #correspond to one time step!!!!!!!!!
    #column 1 = output corresponds to distances
    #column 2 = output corresponds to index of distances
    nmax=len(recsp)-tahead
    Q1=zeros([0,2])
    for j in range(i):
        Q1=vstack([Q1,[linalg.norm(recsp[i,:]-recsp[j,:]),j]])
    for j in range(i+1,nmax):
        Q1=vstack([Q1,[linalg.norm(recsp[i,:]-recsp[j,:]),j]])
    p=argsort(Q1[:,0])
    Q1[:,0]=Q1[p,0]
    Q1[:,1]=Q1[p,1]
    return Q1

STARTTIME=time.time()

#Create the logistic map attractor:
N=5000
A=4
data=zeros(N)
data[0]=pi/10
for i in range(N-1):
    data[i+1]=A*data[i]*(1-data[i])

#Reconstruct a topologically equivalent attractor to the logistic map:
tauI=1 #time lag
dm=3 #embedding dimension
recsp=ReconSp(tauI,dm,data) #time delayed reconstruction
#columns correspond to dimensions,
#rows correspond to time, so each row is a state vector with respect to t
[nmax,dd]=shape(recsp)

#Calculate mean attractor radius:
D=zeros(nmax)
for count in range(nmax):
    D[count]=linalg.norm(recsp[count,:])

print "Mean attractor radius:", mean(D)


eps_tol=0.001 #tolerance level for nearest neighbours
tauS=1 #initial relative time
tauL=50 #final relative time

nst=1-1 #starting index for testing, PYTHON INDEX BEGINS AT 0!
#ned=(nmax-tauL)-1 #ending index for testing, -1 because of 0 indexing!
ned=1000-1
N=range(nst,ned+1,1)

EPNb=[] #list to hold the results

print "Calculating Nearest Neighbors"

for k in xrange(len(N)):
    NbTbl=MDneighbor(N[k],tauL,recsp)
    [ndt,b]=shape(NbTbl)
    epTbl=zeros([0,2])
    for kk in xrange(ndt):
        dist=NbTbl[kk,0]
        theiler_window=0 #Use a Theiler window to prevent autocorrelation error?
        theiler=NbTbl[kk,1]
        if dist<eps_tol and abs(theiler-N[k])>=theiler_window:
            epTbl=vstack([epTbl,NbTbl[kk,:]])
    epList=epTbl[:,1]
    epList=hstack([N[k],epList])
    EPNb.append(epList)
    print "Time step:", k, "No. of NN:", len(epList)-1

print "Calculating S"

StauTbl=zeros([0,2])
for tad in xrange(tauS,tauL+1):
    print tad
    KDTbl=[]
    ntcount=0
    for it in xrange(len(N)):
        mnear=len(EPNb[it])
        if mnear<2:
            continue
        aheadList=ones(mnear)*tad
        epNext=EPNb[it]+aheadList
        KantzDist=0
        for k in xrange(1,mnear):
            KantzDist=KantzDist+linalg.norm(recsp[epNext[0],:]-recsp[epNext[k],:])
        KantzDist=log(KantzDist/(mnear-1))
        KDTbl=hstack([KDTbl,KantzDist])
        ntcount=ntcount+1
    StauList=[tad,sum(KDTbl)/ntcount]
    StauTbl=vstack([StauTbl,StauList])

print "Total points used:", ntcount
print "Points without NN:", len(N)-ntcount

x=StauTbl[:,0] #time delay
y=StauTbl[:,1] #stretching factor

#m=polyfit(StauTbl[0:10,0],StauTbl[0:10,1],1)
#print "Linear Fit:", m

print "Time taken for computation:", time.time()-STARTTIME

plt.plot(x,y,'-o')
plt.xlabel('Tau')
plt.ylabel('S')
plt.grid('on')
plt.title('Lyapunov Exponent for Logistic System')
plt.show()
