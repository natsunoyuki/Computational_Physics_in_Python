import numpy as np
import matplotlib.pyplot as plt
#import time

#This code serves to give a prototype of using the FDTD regime to simulate a 1-dimensional
#laser with 1 port.

#STARTTIME=time.time()

X = 500 #total grid cells to use
source = 100-1 #power source position in cells, -1 for python 0 indexing
T = 100000 #1500000 #time length

#Electromagnetic Pulse Characteristics (CGS units)
c = 1.0 #speed of light
frequency = 1000
T0 = 1.0 / frequency #period of the pulse
tc = 5 * T0 / 2 #time constant
sig = tc / 2 / np.sqrt(2 * np.log(2)) #gaussian pulse width parameter
FWHM = 2 * np.sqrt(2 * np.log(2)) * sig #full wave half maximum
gperp = 35
gpara = gperp / 100
ka = 1000

#Cells and step sizes:
dx = 1e-4 #cell size
dt = dx / (1.0 * c) #Courant No. magic step size of 1 used

#E field and H field constants
c1 = 1/dt**2+gperp/dt
c2 = 2/dt**2-ka**2-gperp**2
c3 = 1/dt**2-gperp/dt
c4 = ka*gperp/2/np.pi
c5 = 1/dt+gpara/2
c6 = 1/dt-gpara/2
c7 = 2*np.pi*gpara/ka
c8 = 1/dt+gperp/2
c9 = -1/dt+gperp/2

#Initialize field vectors in normalised CGS units
Ez = np.zeros(X) #electric field
Hy = np.zeros(X-1) #staggered grid which means H is one less cell than E

#Initialize Absorbing Boundary Conditions
#Ezl.yes=0 #Placeholder Variables for Absorbing Boundary Conditions, Left side
Ezh = 0 #Placeholder Variables for Absorbing Boundary Conditions, Right side

#This part is to change the relative permittivity for the dielectric slab
E = np.ones(X)
E[0] = 0 #perfect reflector
die1 = 1 #left most position of dielectric slab
die2 = 301 #right most position of dielectric slab
n2 = 1.5 #refractive index of the dielectric slab
E[die1:die2] = n2 ** 2 #dielectric constant strength E; n=sqrt(E)
#note that we assume that the dielectric is a perfect magnetic material
#i.e. U0 = 1 for all cells. So we do not need to explicitly have a vector for U0

#Prepare vector to hold electric field at a particular location over the
#entire time frame of the FDTD loop.
yes = np.zeros(T) #field with dielectric
sample = die2-1 #sampling location along the x axis, 0 indexing

D0 = 1.0
D = D0 * np.ones(X)
P = np.zeros(X)
Place = np.zeros(X) #one time step before
Pold = np.zeros(X) #two time steps before

#Main FDTD Loops
for n in range(T):
    #Update the polarization vector
    P[die1:die2]=1/c1*(c2*P[die1:die2]-c3*Pold[die1:die2]-c4*Ez[die1:die2]*D[die1:die2])
    Pold=Place.copy() #DO NOT SIMPLY USE Pold=Place! Use .copy() in python!
    Place=P.copy() #carry the current value of P for two time steps
    
    #Update electric field vector
    Ez[1:X-1]=Ez[1:X-1]-4*np.pi/E[1:X-1]*(P[1:X-1]-Pold[1:X-1])-dt/dx/E[1:X-1]*np.diff(Hy)
    
    #Update Population Inversion Vector
    D[die1:die2]=1/c5*(c6*D[die1:die2]+gpara*D0+c7*Ez[die1:die2]*(c8*P[die1:die2]+c9*Pold[die1:die2]))
    
    #Initiate pulse
    pulse = np.exp((-((n+1)*dt-3*np.sqrt(2)*sig)**2)/(2*sig**2))
    Ez[source] = Ez[source] + pulse
    
    #1st order Mur Boundaries for dielectric
    Ez[X-1] = Ezh+(c*dt-dx)/(c*dt+dx)*(Ez[X-2]-Ez[X-1]);
    Ezh = Ez[X-2]
    
    #update magnetic field vect WITH DIELECTRIC
    Hy=Hy-dt/dx*np.diff(Ez)
    
    #Save the sample data to a vector for export
    yes[n]=Ez[sample]

    if np.remainder(n,10000)==0:
        print('Percent Complete:', n/T*100)

#plot the E and H fields in space
plt.subplot(2,1,1)
plt.plot(Ez)
plt.ylabel('Electric Field')
plt.subplot(2,1,2)
plt.plot(Hy)
plt.ylabel('Magnetic Field')
plt.show()
