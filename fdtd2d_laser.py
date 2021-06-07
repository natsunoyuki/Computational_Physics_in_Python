import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tqdm

# fdtd = fdtd2d_laser()
# fdtd.run()
# fdtd.plot()

class fdtd2d_laser:
    def __init__(self, Nx = 501, Ny = 501, c = 1, dx = 1, dy = 1):
        # Grid attributes
        self.Nx = Nx
        self.Ny = Ny
        self.c = c
        self.dx = dx
        self.dy = dy
        self.dt = min(dx, dy) / np.sqrt(2) / c

        self.x = np.arange(Nx)
        self.y = np.arange(Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Source location
        self.source_x = int(Nx / 2)
        self.source_y = int(Ny / 2)

        # Maxwell-Bloch equation parameters
        self.gperp = 0.01
        self.gpara = self.gperp / 100.0
        self.ka = 0.1
        
        # Magnetic fields H_x and H_y
        self.H_x = np.zeros([Nx, Ny - 1])
        self.H_y = np.zeros([Nx - 1, Ny])
        
        # Electric field E_z
        self.E_z = np.zeros([Nx, Ny])
        
        # Dielectric slab attributes
        self.radius = 200
        self.mask = np.zeros([Nx, Ny])
        for i in range(Nx):
            for j in range(Ny):
                if np.sqrt((i - self.source_x)**2 + (j - self.source_y)**2) < self.radius:
                    self.mask[i, j] = 1
        
        # Permittivity and permeability
        self.n1 = 1.0
        self.n2 = 3.0
        self.E = np.logical_not(self.mask) * self.n1**2 + self.mask * self.n2**2 
        # we assume that the dielectric is a perfect magnetic material with
        # mu = 1 everywhere. So we do not need to explicitly have a vector 
        # for mu. If this condition is not satisfied, then mu must be taken
        # into account as well in the simulation.
        
        # Polarization field P
        self.P = np.zeros([Nx, Ny])
        self.Place = np.zeros([Nx, Ny]) # polarization one time step before
        self.Pold = np.zeros([Nx, Ny]) # polarization two time steps before
        
        # Population inversion D
        self.D0 = 10.0 # pump strength
        self.D = self.mask * self.D0
        
        # Time dependent field for plotting and animation
        self.E_t = []
        
        # Mur absorbing boundaries
        self.E_z_n = self.E_z.copy()     # data for t = n
        self.E_z_n_1 = self.E_z_n.copy() # data for t = n-1
        self.H_x_n = self.H_x.copy()     # data for t = n
        self.H_x_n_1 = self.H_x_n.copy() # data for t = n-1
        self.H_y_n = self.H_x.copy()     # data for t = n
        self.H_y_n_1 = self.H_y_n.copy() # data for t = n-1
        
    def run(self, n_iter = 10000):
        # MB equation constants
        c1 = 1.0 / self.dt ** 2 + self.gperp / self.dt / 2.0
        c2 = 2.0 / self.dt ** 2 - self.ka ** 2 - self.gperp ** 2
        c3 = 1.0 / self.dt ** 2 - self.gperp / self.dt / 2.0
        c4 = self.ka * self.gperp / 2.0 / np.pi
        c5 = 1.0 / self.dt + self.gpara / 2.0
        c6 = 1.0 / self.dt - self.gpara / 2.0
        c7 = 2.0 * np.pi * self.gpara / self.ka
        c8 = 1.0 / self.dt + self.gperp / 2.0
        c9 = -1.0 / self.dt + self.gperp / 2.0
        
        # Mur absorbing boundary constants
        dtdx = np.sqrt(self.dt / self.dx * self.dt / self.dy)
        dtdx_2 = 1 / dtdx + 2 + dtdx
        c_0 = -(1 / dtdx - 2 + dtdx) / dtdx_2
        c_1 = -2 * (dtdx - 1 / dtdx) / dtdx_2
        c_2 = 4 * (dtdx + 1 / dtdx) / dtdx_2
        for n in tqdm.trange(n_iter):
            # Update magnetic fields H_x, H_y
            self.H_x = self.H_x - self.dt / self.dy * (self.E_z[:, 1:] - self.E_z[:, :-1])
            self.H_y = self.H_y + self.dt / self.dx * (self.E_z[1:, :] - self.E_z[:-1, :])

            # Update polarization field P
            self.P = self.mask /c1*(c2*self.P - c3*self.Pold - c4*self.E_z*self.D)
            self.Pold = self.Place.copy() 
            self.Place = self.P.copy() # carry the current value of P for two time steps
            
            # Update electric field E_z
            diff_H_x = self.dt / self.dy / self.E[1:-1, 1:-1] * (self.H_x[1:-1, 1:] - self.H_x[1:-1, :-1])
            diff_H_y = self.dt / self.dx / self.E[1:-1, 1:-1] * (self.H_y[1:, 1:-1] - self.H_y[:-1, 1:-1])
            diff_P = -4 * np.pi / self.E[1:-1, 1:-1] * (self.P[1:-1, 1:-1] - self.Pold[1:-1, 1:-1])
            self.E_z[1:-1, 1:-1] = self.E_z[1:-1, 1:-1] + diff_P + (diff_H_y - diff_H_x)
            
            # Update population inversion D
            self.D = self.mask / c5*(c6*self.D + self.gpara*self.D0 + c7*self.E_z*(c8*self.P + c9*self.Pold))
            
            # Pulse at time step 
            tp = 30
            if n * self.dt <= tp:
                C = (7 / 3) ** 3 * (7 / 4) ** 4
                pulse = C * (n * self.dt / tp) ** 3 * (1 - n * self.dt / tp) ** 4
            else:
                pulse = 0
            self.E_z[self.source_x, self.source_y] = self.E_z[self.source_x, self.source_y] + pulse

            # Mur ABC for top boundary
            self.E_z[0, :] = c_0 * (self.E_z[2, :] + self.E_z_n_1[0, :]) +    \
                             c_1 * (self.E_z_n[0, :] + self.E_z_n[2, :] -    \
                                    self.E_z[1, :] - self.E_z_n_1[1, :]) +    \
                             c_2 * self.E_z_n[1, :] - self.E_z_n_1[2, :]

            # Mur ABC for bottom boundary
            self.E_z[-1, :] = c_0 * (self.E_z[-3, :] + self.E_z_n_1[-1, :]) +    \
                              c_1 * (self.E_z_n[-1, :] + self.E_z_n[-3, :] -    \
                                     self.E_z[-2, :] - self.E_z_n_1[-2, :]) +    \
                              c_2 * self.E_z_n[-2, :] - self.E_z_n_1[-3, :]

            # Mur ABC for left boundary
            self.E_z[:, 0] = c_0 * (self.E_z[:, 2] + self.E_z_n_1[:, 0]) +    \
                             c_1 * (self.E_z_n[:, 0] + self.E_z_n[:, 2] -    \
                                    self.E_z[:, 1] - self.E_z_n_1[:, 1]) +    \
                             c_2 * self.E_z_n[:, 1] - self.E_z_n_1[:, 2]

            # Mur ABC for right boundary
            self.E_z[:, -1] = c_0 * (self.E_z[:, -3] + self.E_z_n_1[:, -1]) +    \
                              c_1 * (self.E_z_n[:, -1] + self.E_z_n[:, -3] -    \
                                     self.E_z[:, -2] - self.E_z_n_1[:, -2]) +    \
                              c_2 * self.E_z_n[:, -2] - self.E_z_n_1[:, -3]

            # Store magnetic and electric fields for ABC at time step n
            self.H_x_n_1 = self.H_x_n.copy() # data for t = n-1
            self.H_x_n = self.H_x.copy()     # data for t = n

            self.H_y_n_1 = self.H_y_n.copy() # data for t = n-1
            self.H_y_n = self.H_y.copy()     # data for t = n

            self.E_z_n_1 = self.E_z_n.copy() # data for t = n-1
            self.E_z_n = self.E_z.copy()     # data for t = n

            #self.E_t.append(self.E_z[self.source_x, self.source_y])
            self.E_t.append(self.E_z.copy())
            if len(self.E_t) > 500:
                del self.E_t[0]
            
    def plot(self, i = -1):
        plt.figure(figsize = (5, 5))
        #plt.pcolormesh(self.x, self.y, self.E_z, shading = "auto", cmap = "gray")
        plt.pcolormesh(self.x, self.y, self.E_t[i], 
                       #vmin = np.min(self.E_t), vmax = np.max(self.E_t), 
                       shading = "auto", cmap = "bwr")
        circle = plt.Circle((self.source_x, self.source_y), self.radius, color = "k", fill = False)
        plt.gca().add_patch(circle)
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.axis("equal")
        #plt.colorbar()
        plt.show()
            
    def animate(self, file_dir = "fdtd_2d_animation.gif", N = 500):
        # animate self.Et as a .gif file.
        # N: number of total steps to save as .gif animation.
        E_t = self.E_t[-N:]

        fig, ax = plt.subplots(figsize = (5, 5))
        cax = ax.pcolormesh(self.x, self.y, E_t[0], 
                            vmin = np.min(E_t), vmax = np.max(E_t), 
                            shading = "auto", cmap = "bwr")
        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        
        circle = plt.Circle((self.source_x, self.source_y), self.radius, color = "k", fill = False)
        plt.gca().add_patch(circle)

        def animate(i):
            cax.set_array(E_t[i].flatten())

        anim = FuncAnimation(fig, animate, interval = 50, frames = len(E_t) - 1)

        #plt.show()

        anim.save(file_dir, writer = "pillow")
