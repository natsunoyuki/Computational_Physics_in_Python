import numpy as np
import matplotlib.pyplot as plt

# Usage:
# fdtd = fdtd1d()
# fdtd.run()
# fdtd.plot()

class fdtd1d(object):
    # This code serves to give a demonstration of using the FDTD method to 
    # simulate a 1-dimensional laser with 1 port on the right side, and a 
    # perfectly reflecting mirror on the left side of the cavity.
    def __init__(self, X = 500, dx = 1e-4, T = 100000, source = 99,
                 c = 1.0, frequency = 1000, 
                 gperp = 35, ka = 1000, 
                 die1 = 1, die2 = 301, n2 = 1.5, D0 = 1.0):
        
        # Physical simulation parameters
        self.X = X # number of grid cells in the x direction
        self.dx = dx # cell size
        self.T = T # number of time steps
        self.source = source # power source location in cells.
        
        # Lasing parameters in CGS units:
        self.c = c # speed of light
        self.frequency = frequency # frequency
        self.T0 = 1.0 / frequency # period
        self.tc = 5 * self.T0 / 2.0 # time constant
        self.sig = self.tc / 2.0 / np.sqrt(2 * np.log(2)) # gaussian width
        self.FWHM = 2 * np.sqrt(2 * np.log(2)) * self.sig # FWHM of pulse
        self.gperp = gperp # g_perpendicular lasing parameter
        self.gpara = gperp / 100.0 # p_parallel lasing parameter
        self.ka = ka # ka lasing parameter
        
        self.dt = self.dx / (1.0 * self.c) # Courant No. magic step size = 1
        
        # E field and H field constants
        self.c1 = 1.0 / self.dt ** 2 + self.gperp / self.dt
        self.c2 = 2.0 / self.dt ** 2 - self.ka**2 - self.gperp ** 2
        self.c3 = 1.0 / self.dt ** 2 - self.gperp / self.dt
        self.c4 = self.ka * self.gperp / 2.0 / np.pi
        self.c5 = 1.0 / self.dt + self.gpara / 2.0
        self.c6 = 1.0 / self.dt - self.gpara / 2.0
        self.c7 = 2.0 * np.pi * self.gpara / ka
        self.c8 = 1.0 / self.dt + self.gperp / 2.0
        self.c9 = -1.0 / self.dt + self.gperp / 2.0
              
        # Initialize field vectors in normalised CGS units
        # electric field
        self.Ez = np.zeros(self.X) 
        # staggered grid which means H is one fewer grid cell than E
        self.Hy = np.zeros(self.X - 1) 
        
        # Initialize Absorbing Boundary Conditions
        # Placeholder Variables for Absorbing Boundary Conditions, Left side
        #self.Ezl.yes=0
        # Placeholder Variables for Absorbing Boundary Conditions, Right side
        self.Ezh = 0 
        
        # Change the relative permittivity for the dielectric slab
        self.E = np.ones(self.X)
        self.E[0] = 0 # perfect reflector on the left side
        self.die1 = die1 # left most position of dielectric slab
        self.die2 = die2 # right most position of dielectric slab
        self.n2 = n2 # refractive index of the dielectric slab
        self.E[die1:die2] = n2 ** 2 # dielectric constant strength E
        # we assume that the dielectric is a perfect magnetic material
        # U0 = 1 for all cells. So we do not need to explicitly have a vector 
        # for U0
        
        # Prepare vector to hold electric field at a particular location over 
        # the entire time frame of the FDTD loop.
        self.Et = np.zeros(self.T) # time based field measured at sample
        self.sample = die2 - 1 # sampling location along the x axis, 0 indexing
        
        self.D0 = D0 # polarization constant
        self.D = self.D0 * np.ones(X)
        self.P = np.zeros(X) # polarization vector
        self.Place = np.zeros(X) # polarization one time step before
        self.Pold = np.zeros(X) # polarization two time steps before
        
    def get_fields(self):
        return self.Ez, self.Hy
    
    def get_et(self):
        return self.Et

    def run(self):
        print("Running FDTD...")
        # Main FDTD Loops
        for n in range(self.T):
            # Update the polarization vector
            self.P[self.die1:self.die2] = 1.0 / self.c1 * (self.c2 * self.P[self.die1:self.die2] - self.c3 * self.Pold[self.die1:self.die2] - self.c4 * self.Ez[self.die1:self.die2] * self.D[self.die1:self.die2])
            self.Pold = self.Place.copy() # DO NOT SIMPLY USE Pold=Place! Use .copy() in python!
            self.Place = self.P.copy() # carry the current value of P for two time steps
            
            # Update electric field vector
            self.Ez[1:self.X-1] = self.Ez[1:self.X-1] - 4 * np.pi / self.E[1:self.X-1] * (self.P[1:self.X-1] - self.Pold[1:self.X-1]) - self.dt / self.dx / self.E[1:self.X-1] * np.diff(self.Hy)
            
            # Update Population Inversion Vector
            self.D[self.die1:self.die2] = 1.0 / self.c5 * (self.c6 * self.D[self.die1:self.die2] + self.gpara * self.D0 + self.c7 * self.Ez[self.die1:self.die2] * (self.c8 * self.P[self.die1:self.die2] + self.c9 * self.Pold[self.die1:self.die2]))
            
            # Initiate EM pulse
            pulse = np.exp((-((n+1) * self.dt - 3 * np.sqrt(2) * self.sig)**2) / (2*self.sig**2))
            self.Ez[self.source] = self.Ez[self.source] + pulse
            
            # 1st order Mur Boundaries for dielectric to absorb the outgoing field
            self.Ez[self.X-1] = self.Ezh+(self.c * self.dt - self.dx) / (self.c * self.dt + self.dx) * (self.Ez[self.X-2] - self.Ez[self.X-1])
            self.Ezh = self.Ez[self.X-2]
            
            # update magnetic field vect WITH DIELECTRIC
            self.Hy = self.Hy - self.dt / self.dx * np.diff(self.Ez)
            
            # Save the sample data to a vector for export
            self.Et[n] = self.Ez[self.sample]
        
            if np.remainder(n, int(self.T / 10)) == 0:
                print('Percent Complete: {:.0f}%'.format(n / self.T * 100))
                
        print("Run complete!")

    def plot(self):
        # plot the Ez and Hy fields in space
        plt.figure(figsize = (10, 5))
        plt.subplot(2,1,1)
        plt.plot(self.Ez, 'b')
        plt.axvline(x = self.die1, color = 'k')
        plt.axvline(x = self.die2, color = 'k', linestyle = '-.')
        plt.ylabel('Ez')
        plt.grid('on')
        plt.subplot(2,1,2)
        plt.plot(self.Hy, 'r')
        plt.axvline(x = self.die1, color = 'k')
        plt.axvline(x = self.die2, color = 'k', linestyle = '-.')
        plt.ylabel('Hy')
        plt.xlabel("x")
        plt.grid('on')
        plt.show()
        
    def plot_et(self):
        # plot the time based solution Et
        plt.figure(figsize = (10, 5))
        plt.plot(self.Et)
        plt.xlabel("t")
        plt.ylabel("Ez(t)")
        plt.grid('on')
        plt.show()