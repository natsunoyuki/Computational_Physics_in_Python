import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# FDTD 2D, TEZ polarization simulation.
# The magnetic field oscillates only in the z direction, while the electric field
# oscillates only in the x and y directions. A staggered grid in space is used to 
# simulate H_z, E_x and E_y:
#     E_y(m, n+1/2) H_z(m+1/2, n+1/2) 
#                   E_x(m+1/2, n) 

class fdtd2d_tez:
    def __init__(self, Nx = 101, Ny = 101, c = 1, dx = 1, dy = 1):
        self.Nx = Nx
        self.Ny = Ny
        self.c = c
        self.dx = dx
        self.dy = dy
        self.dt = min(dx, dy) / np.sqrt(2) / c

        self.x = np.arange(Nx)
        self.y = np.arange(Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.source_x = int(Nx / 2)
        self.source_y = int(Ny / 2)

        self.H_z = np.zeros([Nx - 1, Ny - 1])
        self.E_x = np.zeros([Nx - 1, Ny])
        self.E_y = np.zeros([Nx, Ny - 1])

        self.H_z_t = []
        self.E_x_t = []
        self.E_y_t = []
        
        # Fields at time n, n-1 for Mur ABC
        self.H_z_n = self.H_z.copy()
        self.H_z_n_1 = self.H_z_n.copy()
        self.E_x_n = self.E_x.copy()
        self.E_x_n_1 = self.E_x_n.copy()
        self.E_y_n = self.E_y.copy()
        self.E_y_n_1 = self.E_y_n.copy()
        
    def run(self, n_iter = 150):
        # Mur ABC coefficients
        dtdx = np.sqrt(self.dt / self.dx * self.dt / self.dy)
        dtdx_2 = 1 / dtdx + 2 + dtdx
        c_0 = -(1 / dtdx - 2 + dtdx) / dtdx_2
        c_1 = -2 * (dtdx - 1 / dtdx) / dtdx_2
        c_2 = 4 * (dtdx + 1 / dtdx) / dtdx_2
        
        # FDTD Loop
        for n in range(n_iter):
            # Update magnetic field at time step n+1/2
            diff_E_x = self.dt / self.dy * (self.E_x[:, 1:] - self.E_x[:, :-1])
            diff_E_y = self.dt / self.dx * (self.E_y[1:, :] - self.E_y[:-1, :])
            self.H_z = self.H_z - (diff_E_y - diff_E_x)
            
            # Update electric fields at time step n+1
            self.E_x[:, 1:-1] = self.E_x[:, 1:-1] + self.dt / self.dy * (self.H_z[:, 1:] - self.H_z[:, :-1])
            self.E_y[1:-1, :] = self.E_y[1:-1, :] - self.dt / self.dx * (self.H_z[1:, :] - self.H_z[:-1, :])
            
            # Pulse at time step n+1
            tp = 30
            if n * self.dt <= tp:
                C = (7 / 3) ** 3 * (7 / 4) ** 4
                pulse = C * (n * self.dt / tp) ** 3 * (1 - n * self.dt / tp) ** 4
            else:
                pulse = 0
                
            self.H_z[self.source_x, self.source_y] = self.H_z[self.source_x, self.source_y] + pulse
            #self.E_x[self.source_x, self.source_y] = self.E_x[self.source_x, self.source_y] + pulse
            #self.E_y[self.source_x, self.source_y] = self.E_y[self.source_x, self.source_y] + pulse
            
            # Mur ABC for left boundary
            self.E_y[0, :] = c_0 * (self.E_y[2, :] + self.E_y_n_1[0, :]) +    \
                             c_1 * (self.E_y_n[0, :] + self.E_y_n[2, :] -    \
                                    self.E_y[1, :] - self.E_y_n_1[1, :]) +    \
                             c_2 * self.E_y_n[1, :] - self.E_y_n_1[2, :]

            # Mur ABC for right boundary
            self.E_y[-1, :] = c_0 * (self.E_y[-3, :] + self.E_y_n_1[-1, :]) +    \
                              c_1 * (self.E_y_n[-1, :] + self.E_y_n[-3, :] -    \
                                     self.E_y[-2, :] - self.E_y_n_1[-2, :]) +    \
                              c_2 * self.E_y_n[-2, :] - self.E_y_n_1[-3, :]

            # Mur ABC for bottom boundary
            self.E_x[:, 0] = c_0 * (self.E_x[:, 2] + self.E_x_n_1[:, 0]) +    \
                             c_1 * (self.E_x_n[:, 0] + self.E_x_n[:, 2] -    \
                                    self.E_x[:, 1] - self.E_x_n_1[:, 1]) +    \
                             c_2 * self.E_x_n[:, 1] - self.E_x_n_1[:, 2]

            # Mur ABC for right boundary
            self.E_x[:, -1] = c_0 * (self.E_x[:, -3] + self.E_x_n_1[:, -1]) +    \
                              c_1 * (self.E_x_n[:, -1] + self.E_x_n[:, -3] -    \
                                     self.E_x[:, -2] - self.E_x_n_1[:, -2]) +    \
                              c_2 * self.E_x_n[:, -2] - self.E_x_n_1[:, -3]

            # Store magnetic and electric fields for ABC at time step n
            self.E_x_n_1 = self.E_x_n.copy() # data for t = n-1
            self.E_x_n = self.E_x.copy()     # data for t = n

            self.E_y_n_1 = self.E_y_n.copy() # data for t = n-1
            self.E_y_n = self.E_y.copy()     # data for t = n

            self.H_z_n_1 = self.H_z_n.copy() # data for t = n-1
            self.H_z_n = self.H_z.copy()     # data for t = n

            self.H_z_t.append(self.H_z.copy())
            if len(self.H_z_t) > 500:
                del self.H_z_t[0]
                
            self.E_x_t.append(self.E_x.copy())
            if len(self.E_x_t) > 500:
                del self.E_x_t[0]
                
            self.E_y_t.append(self.E_y.copy())
            if len(self.E_y_t) > 500:
                del self.E_y_t[0]
            
    def plot_H(self, i = 70):
        if i >= len(self.H_z_t):
            i = len(self.H_z_t) - 1
        plt.figure(figsize = (5, 5))
        plt.pcolormesh(self.X[1:, 1:], self.Y[1:, 1:], self.H_z_t[i].T, 
                       #vmin = np.min(self.E_t), vmax = np.max(self.E_t), 
                       shading = "auto", cmap = "bwr")
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.axis("equal")
        plt.show()
        
    def plot_E(self, i = 70):
        if i >= len(self.E_x_t):
            i = len(self.E_x_t) - 1
        plt.figure(figsize = (10, 5))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(self.X[:, 1:], self.Y[:, 1:], self.E_x_t[i].T, 
                       #vmin = np.min(self.E_t), vmax = np.max(self.E_t), 
                       shading = "auto", cmap = "bwr")
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.axis("equal")
        plt.subplot(1, 2, 2)
        plt.pcolormesh(self.X[1:, :], self.Y[1:, :], self.E_y_t[i].T, 
                       #vmin = np.min(self.E_t), vmax = np.max(self.E_t), 
                       shading = "auto", cmap = "bwr")
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.axis("equal")
        plt.show()
            
    def animate_H(self, file_dir = "fdtd_2d_animation.gif", N = 500):
        # animate self.H_t as a .gif file.
        # N: number of total steps to save as .gif animation.
        H_z_t = self.H_z_t[-N:]

        fig, ax = plt.subplots(figsize = (5, 5))
        cax = ax.pcolormesh(self.X[1:, 1:], self.Y[1:, 1:], H_z_t[0].T, 
                            vmin = np.min(H_z_t), vmax = 0.1 * np.max(H_z_t), 
                            shading = "auto", cmap = "gray")
        plt.axis("equal")
        plt.grid(True)

        def animate(i):
            cax.set_array(H_z_t[i].T.flatten())

        anim = FuncAnimation(fig, animate, interval = 50, frames = len(H_z_t) - 1)
        anim.save(file_dir, writer = "pillow")
        plt.show()