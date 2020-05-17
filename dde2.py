import numpy as np
import matplotlib.pyplot as plt
import time

#This is a code to simulate the time forward modelling of a delay difference equation.

#This code solves the delay difference equations proposed by:

#M. Nakanishi, T. Koyaguchi, "A stability analysis of a conduit flow model for lave dome eruptions",
#Journal of Volcanology and Geothermal Research, 178, 46-57, 2008.

starttime=time.time() #log start time

#functions to hold the delay difference equation:
def less_than_one(x,t,delayx):
    ydot=np.zeros(len(x))
    ydot[0]=Qin-x[1] #pressure
    ydot[1]=x[1]/x[0]*(Qin-x[1]+(mu-1)*(x[1]-delayx)*x[1]) #flux
    return ydot
    
def more_than_one(x,t,delayx):
    ydot=np.zeros(len(x))
    ydot[0]=Qin-x[1] #pressure
    ydot[1]=Qin-x[1] #flux
    return ydot

#Runge-Kutta-4 regime to solve differential equations:
def rk4(t0,t1,y0,ydot_fun,params):
    dt=t1-t0
    k1=(dt)*ydot_fun(y0,t0,params)
    k2=(dt)*ydot_fun(y0+0.5*k1,t0+(dt)*0.5,params)
    k3=(dt)*ydot_fun(y0+0.5*k2,t0+(dt)*0.5,params)
    k4=(dt)*ydot_fun(y0+k3,t0+(dt),params)
    y1=y0+(k1+2*k2+2*k3+k4)/6.0
    return y1

#dde parameters:

Qin=0.8
mu=10
y0=np.array([1.5,1.5]) #initial values 1.1,1.1 [P,Q]
tstar=1 #interesting values to use: 1,0.2
dt=tstar/20000 #set time step size
delayindex=np.int(tstar/dt) #for array calling
t=np.arange(-tstar,30+dt,dt) #length of time of the simulation
Y=np.zeros([len(t),len(y0)])
Y[0,:]=y0 #set initial conditions for the simulation
#Create data history:
for i in range(delayindex):
    y0[0]=y0[0]-1./delayindex #P history
    y0[1]=y0[1]-1./delayindex #Q history
    Y[i+1,:]=y0
#Y[0:delayindex+1]=y0
xx=np.zeros(len(t)-1-delayindex)

#Actual RK4 loop to solve the delay differential equation:
for i in range(len(t)-1-delayindex):
    params=Y[i,1]
    T=t[i:i+delayindex+1]
    Q=Y[i:i+delayindex+1,1]
    xx[i]=np.trapz(Q,T) #check the value of the integral of Q w.r.t. T
    if xx[i]>1:
        Y[i+1+delayindex,:]=rk4(t[i+delayindex],t[i+1+delayindex],Y[i+delayindex,:],more_than_one,params)
    else:
        Y[i+1+delayindex,:]=rk4(t[i+delayindex],t[i+1+delayindex],Y[i+delayindex,:],less_than_one,params)
    
X=Y[delayindex:] #remove the (artificial) history from the data arrays
T=t[delayindex:]

endtime=time.time()-starttime
print("Time elapsed: {:.2f}s".format(endtime)) #print time elapsed

plt.plot(X[:,1],X[:,0]) #plot the simulated data
plt.show() #show the plots on screen
