import numpy as np
import matplotlib.pyplot as plt
import tqdm

# Usage:
# fdtd = fdtd1d()
# fdtd.run()
# fdtd.plot()

class fdtd1d_laser(object):
    def __init__(self, Nx = 201, dx = 1e-3, c = 1, source = 100, sample = 100, n_iter = 150):      
        # Grid properties:
        # 1. Number of grid cells
        self.Nx = Nx
        # 2. Grid cell size
        self.dx = dx
        # 3. Speed of light
        self.c = c
        # 4. Time step size (obeying Courant number)
        self.dt = dx / (1.0 * self.c)

        # Source properties:
        # 1. Source location array index
        self.source = source
        # 2. Source frequency, period etc.:
        frequency = 1 / (self.dt * 10)
        T0 = 1.0 / frequency
        tc = 5 * T0 / 2
        self.sig = tc / 2 / np.sqrt(2 * np.log(2))

        # Initialize grid:
        # 1. Electric field grid
        self.E_y = np.zeros(Nx)
        # 2. Magnetic field grid
        self.H_z = np.zeros(Nx - 1) 
        # 3. Physical grid (for visualization)
        self.x = np.arange(0, Nx, 1)
        self.Dx = np.arange(1, Nx, 1)

        # Mur absorbing boundary conditions (ABC)
        # ABC for right side 
        self.E_y_h = 0 
        # ABC for left side
        self.E_y_l = 0

        # Record the time dependent electric field at sample.
        self.sample = sample
        self.E_t = []
        
        # Physical grid for plotting
        self.x = np.arange(0, X, 1)
        self.Dx = np.arange(1, X, 1)
        
        # Total number of time steps to run
        self.n_iter = 150

    def run(self):
        # Main FDTD Loops
        dt = self.dt
        dx = self.dx
        c = self.c
        sample = self.sample
        sig = self.sig
        source = self.source
        
        for n in tqdm.trange(self.n_iter):
            # Update magnetic field
            self.H_z = self.H_z - dt / dx * (self.E_y[1:] - self.E_y[:-1])     
    
            # Update electric field
            self.E_y[1:-1] = self.E_y[1:-1] - dt / dx * (self.H_z[1:] - self.H_z[:-1])
        
            # Initiate source
            pulse = np.exp((-((n+1) * dt - 3 * np.sqrt(2) * sig)**2) / (2 * sig**2))
            self.E_y[source] = self.E_y[source] + pulse
         
            # Mur ABC for right side   
            self.E_y[-1] = self.E_y_h + (c * dt - dx) / (c * dt + dx) * (self.E_y[-2] - self.E_y[-1])
            self.E_y_h = self.E_y[-2]
    
            # Mur ABC for left side
            self.E_y[0] = self.E_y_l + (c * dt - dx) / (c * dt + dx) * (self.E_y[1] - self.E_y[0])
            self.E_y_l = self.E_y[1]
            
            self.E_t.append(self.E_y[sample])
        
    def plot(self):
        # plot the E_y and H_z fields in space
        plt.figure(figsize = (10, 5))
        plt.plot(self.x, self.E_y)
        plt.plot(self.Dx, self.H_z)
        plt.ylabel('E_y, H_y')
        plt.grid('on')
        plt.xlabel("x")
        plt.legend(["E_y", "H_z"])
        plt.show()
        
    def plot_et(self, t0 = None, t1 = None):
        # plot the time dependent electric field solution Et
        if t0 is None:
            t0 = 0
        if t1 is None:
            t1 = len(self.E_t)
        if t1 > len(self.E_t):
            t1 = len(self.E_t)
        plt.figure(figsize = (10, 5))
        plt.plot(self.E_t[t0:t1], 'red')
        plt.xlabel("t")
        plt.ylabel("Ez(t)")
        plt.grid('on')
        plt.show()