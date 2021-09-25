import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# FDTD 2D, TMZ polarization simulation.
# The electric field oscillates only in the z direction, while the magnetic field
# oscillates only in the x and y directions. A staggered grid in space is used to 
# simulate E_z, H_x and H_y:
#     H_x(m, n+1/2) 
#     E_z(m, n)     H_y(m+1/2, n)

class fdtd2d_tmz:
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

        self.E_z = np.zeros([Nx, Ny])
        self.H_x = np.zeros([Nx, Ny - 1])
        self.H_y = np.zeros([Nx - 1, Ny])

        self.E_z_t = []
        self.H_x_t = []
        self.H_y_t = []

        # Fields at time n, n-1 for Mur ABC
        self.E_z_n = self.E_z.copy()
        self.E_z_n_1 = self.E_z_n.copy()
        self.H_x_n = self.H_x.copy()
        self.H_x_n_1 = self.H_x_n.copy()
        self.H_y_n = self.H_x.copy()
        self.H_y_n_1 = self.H_y_n.copy()
        
    def run(self, n_iter = 150):
        # Mur ABC coefficients
        dtdx = np.sqrt(self.dt / self.dx * self.dt / self.dy)
        dtdx_2 = 1 / dtdx + 2 + dtdx
        c_0 = -(1 / dtdx - 2 + dtdx) / dtdx_2
        c_1 = -2 * (dtdx - 1 / dtdx) / dtdx_2
        c_2 = 4 * (dtdx + 1 / dtdx) / dtdx_2
        
        # FDTD Loop
        for n in range(n_iter):
            # Update magnetic fields at time step n+1/2
            self.H_x = self.H_x - self.dt / self.dy * (self.E_z[:, 1:] - self.E_z[:, :-1])
            self.H_y = self.H_y + self.dt / self.dx * (self.E_z[1:, :] - self.E_z[:-1, :])

            # Update electric field at time step n+1
            diff_H_x = self.dt / self.dy * (self.H_x[1:-1, 1:] - self.H_x[1:-1, :-1])
            diff_H_y = self.dt / self.dx * (self.H_y[1:, 1:-1] - self.H_y[:-1, 1:-1])
            self.E_z[1:-1, 1:-1] = self.E_z[1:-1, 1:-1] + (diff_H_y - diff_H_x)

            # Pulse at time step n+1
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
            self.E_z_t.append(self.E_z.copy())
            self.H_x_t.append(self.H_x.copy())
            self.H_y_t.append(self.H_y.copy())
            if len(self.E_z_t) > 500:
                del self.E_z_t[0]
                del self.H_x_t[0]
                del self.H_y_t[0]
                            
    def plot_E(self, i = 70):
        if i >= len(self.E_z_t):
            i = len(self.E_z_t) - 1
        plt.figure(figsize = (5, 5))
        plt.pcolormesh(self.x, self.y, self.E_z_t[i].T, 
                       #vmin = np.min(self.E_t), vmax = np.max(self.E_t), 
                       shading = "auto", cmap = "bwr")
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()
        
    def plot_H(self, i = 70):
        if i >= len(self.H_x_t):
            i = len(self.H_x_t) - 1
        plt.figure(figsize = (10, 5))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(self.X[1:, :], self.Y[1:, :], self.H_x_t[i].T, 
                       #vmin = np.min(self.E_t), vmax = np.max(self.E_t), 
                       shading = "auto", cmap = "bwr")
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.pcolormesh(self.X[:, 1:], self.Y[:, 1:], self.H_y_t[i].T, 
                       #vmin = np.min(self.E_t), vmax = np.max(self.E_t), 
                       shading = "auto", cmap = "bwr")
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()
        
    def animate_E(self, file_dir = "fdtd_2d_E_animation.gif", N = 500):
        # animate self.E_z_t as a .gif file.
        # N: number of total steps to save as .gif animation.
        E_z_t = self.E_z_t[-N:]

        fig, ax = plt.subplots(figsize = (5, 5))
        cax = ax.pcolormesh(self.x, self.y, E_z_t[0].T, 
                            vmin = np.min(E_z_t), vmax = np.max(E_z_t), 
                            shading = "auto", cmap = "gray")
        plt.axis("equal")
        plt.grid(True)

        def animate(i):
            cax.set_array(E_z_t[i].T.flatten())

        anim = FuncAnimation(fig, animate, interval = 50, frames = len(E_z_t) - 1)
        anim.save(file_dir, writer = "pillow")
        plt.show()
        
    def animate_H(self, file_dir = "fdtd_2d_H_animation.gif", N = 500):
        # animate self.H_x,y_t as a .gif file.
        # N: number of total steps to save as .gif animation.
        H_x_t = self.H_x_t[-N:]
        H_y_t = self.H_y_t[-N:]
        
        fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))
        cax1 = ax1.pcolormesh(self.X[1:, :], self.Y[1:, :], H_x_t[0].T, 
                              vmin = np.min(H_x_t), vmax = 0.1 * np.max(H_x_t), 
                              shading = "auto", cmap = "gray")
        ax1.axis("equal")
        ax1.grid(True)
        
        cax2 = ax2.pcolormesh(self.X[:, 1:], self.Y[:, 1:], H_y_t[0].T, 
                              vmin = np.min(H_y_t), vmax = 0.1 * np.max(H_y_t), 
                              shading = "auto", cmap = "gray")
        ax2.axis("equal")
        ax2.grid(True)

        def animate(i):
            cax1.set_array(H_x_t[i].T.flatten())
            cax2.set_array(H_y_t[i].T.flatten())

        anim = FuncAnimation(fig, animate, interval = 50, frames = len(H_x_t) - 1)
        anim.save(file_dir, writer = "pillow")
        plt.show()