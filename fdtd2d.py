import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class fdtd2d:
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

        self.E_t = []

        self.E_z_n = self.E_z.copy()
        self.E_z_n_1 = self.E_z_n.copy()
        self.H_x_n = self.H_x.copy()
        self.H_x_n_1 = self.H_x_n.copy()
        self.H_y_n = self.H_x.copy()
        self.H_y_n_1 = self.H_y_n.copy()
        
    def run(self, n_iter = 150):
        dtdx = np.sqrt(self.dt / self.dx * self.dt / self.dy)
        dtdx_2 = 1 / dtdx + 2 + dtdx
        c_0 = -(1 / dtdx - 2 + dtdx) / dtdx_2
        c_1 = -2 * (dtdx - 1 / dtdx) / dtdx_2
        c_2 = 4 * (dtdx + 1 / dtdx) / dtdx_2
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
            self.E_t.append(self.E_z.copy())
            
    def plot(self, i = 70):
        plt.figure(figsize = (5, 5))
        #plt.pcolormesh(self.x, self.y, self.E_z, shading = "auto", cmap = "gray")
        plt.pcolormesh(self.x, self.y, self.E_t[i], 
                       #vmin = np.min(self.E_t), vmax = np.max(self.E_t), 
                       shading = "auto", cmap = "bwr")
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
        plt.grid(True)

        def animate(i):
            cax.set_array(E_t[i].flatten())

        anim = FuncAnimation(fig, animate, interval = 50, frames = len(E_t) - 1)

        #plt.show()

        anim.save(file_dir, writer = "pillow")