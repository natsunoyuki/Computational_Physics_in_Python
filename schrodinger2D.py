import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as sla
 
def schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, Vfun2D, neigs, E0=0.0, findpsi=False):
    x = np.linspace(xmin, xmax, Nx)  
    dx = x[1] - x[0]  
    y = np.linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]
    M = Nx * Ny 

    V = Vfun2D(x, y)

    Hx = sparse.lil_matrix(2 * np.eye(Nx))
    for i in range(Nx - 1):
        Hx[i, i + 1] = -1
        Hx[i + 1, i] = -1
    Hx = Hx / (dx ** 2)
     
    Hy = sparse.lil_matrix(2 * np.eye(Ny))
    for i in range(Ny - 1):
        Hy[i, i + 1] = -1
        Hy[i + 1, i] = -1
    Hy = Hy / (dy ** 2)

    Ix = sparse.lil_matrix(np.eye(Nx))
    Iy = sparse.lil_matrix(np.eye(Ny))
    H = sparse.kron(Iy, Hx) + sparse.kron(Hy, Ix)  

    H = H.tolil()

    for i in range(M):
        H[i, i] = H[i, i] + V[i]    

    H = H.tocsc()  

    [evl, evt] = sla.eigs(H, k=neigs, sigma=E0)
            
    if findpsi == False:
        return evl
    else: 
        return evl, evt, x, y
    
def Vfun2D(X, Y):
    Nx = len(X)
    Ny = len(Y)
    M = Nx * Ny
    V = np.zeros([M, 1])
    vindex = 0
    for i in range(Nx):
        for j in range(Ny):
            V[vindex] = X[i] ** 2 + Y[j] ** 2
            vindex = vindex + 1
    return V

def stadium_wavefunctions_plot(R=1, L=2, E0=1000):
    
    xmin = -0.5 * L - R
    xmax = 0.5 * L + R
    ymin = -R
    ymax = R
    print(xmin, xmax, ymin, ymax)

    Nx = 250

    Ny = int(Nx * 2 * R / (2.0 * R + L))
    print(Nx, Ny)
    neigs = 6
    
    def Vfun2D(X, Y):
        Nx = len(X)
        Ny = len(Y)
        [x, y] = np.meshgrid(X, Y)
        F = np.zeros([Ny, Nx])

        for i in range(Ny):
            for j in range(Nx):
                if abs(Y[i]) == R or abs(X[j]) == R + 0.5 * L:
                    F[i, j] = 10000
                if (abs(X[j]) - 0.5 * L) > 0 and np.sqrt((abs(X[j]) - 0.5 * L) ** 2 + Y[i] ** 2) >= R:
                    F[i, j] = 10000

        V = np.zeros([Nx * Ny, 1])
        vindex = 0
        for i in range(Ny):
            for j in range(Nx):
                V[vindex] = F[i, j]
                vindex = vindex + 1                      
        return V
    
    V = Vfun2D(np.linspace(xmin, xmax, Nx), np.linspace(ymin, ymax, Ny))
    F = np.zeros([Ny, Nx])
    vindex = 0
    for i in range(Ny):
        for j in range(Nx):
            F[i, j] = V[vindex]  
    F = np.flipud(F)

    
    H = schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, Vfun2D, neigs, E0, findpsi=True)
    evt = H[1]  

    plt.figure(figsize=(8,8))
    G = np.zeros([Ny, Nx])
    for n in range(6):
        psi = evt[:, n]  
        vindex = 0
        for i in range(Ny):
            for j in range(Nx):
                G[i, j] = psi[vindex]
                vindex = vindex + 1
        G = np.flipud(G)        
        plt.subplot(2, 3, n + 1)    
        plt.pcolormesh(G.T,cmap='jet')
        plt.axis('equal')
        plt.axis('off')
    
    plt.show()
   